"""Does `xgb_classifier` reproduce the model it claims to compile in?

The `xgb-classifier` crate transpiles `models/per_request/xgb_transfer_bytes.json`
to Rust with m2cgen and ports the `engineer_features` pipeline that feeds it, so
that the shipped model can be priced against `llm-classifier`'s lookup table on
what it costs a browser to ship. That comparison is only worth anything if the
crate really is the same predictor, which is what these tests establish: the 80
features against `engineer_features`' own, then the prediction against
`model.predict`.

The crate is not quite *identical* to `xgboost_shipped` here, because the
harness cannot hand it the query string the log measured.
`INTERFACE_LOSSES` names every feature that costs, and
`test_interface_loss_is_confined_to_known_features` fails if a new one appears
— so the gap stays enumerated rather than drifting. It is down to one entry,
and that one belongs to this harness rather than to the interface: once
`estimate_resources` took the initiator and the method and `RequestContext`
gained a JSON variant, the only thing left wrong is a query the log never
kept.

Every test runs once per dataset in `conftest.DATASETS`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import solutions as solutions_mod
from journey import ROOT, Split
from train_multi_target import engineer_features
from url_embeddings import URLEmbedder

# How much of `engineer_features`' answer the crate is allowed to miss by.
# Two ulp of float32, which is what its arithmetic accumulates: the embedding
# runs in float32 in scipy's order, so the low bit of a component is not
# reproducible and is not worth reproducing. Everything else matches exactly.
FEATURE_TOLERANCE = 2 * np.finfo(np.float32).eps

# Rows to compare. A few hundred covers every code path in the feature port
# — the fallback chain, an absent extension, an empty path — and each one
# costs a Python feature frame.
SAMPLE_ROWS = 400
SAMPLE_SEED = 1

# Features that come out different from the log's own, and why. Keys are the
# booster's feature names.
#
# Nothing on this list is the interface's fault any more — every entry is this
# harness reconstructing a URL it did not keep, and would cost a browser
# holding the real URL nothing.
#
# What is *not* here: `init_script`, `init_parser`, `init_other` and `is_post`,
# which `estimate_resources` takes as arguments; and `rt_other` and
# `domain_type_median`, which were wrong for `json` requests until
# `RequestContext` gained a JSON variant. Those five were the whole of the
# interface's cost — see the crate's README.
INTERFACE_LOSSES = {
    # `estimator_urls` rebuilds a query matching the recorded length and
    # parameter count, so `num_query_params` is exact; what it cannot match is
    # a `?` that the length budget rounds away.
    "has_query_params": "the harness rebuilds the query out of filler",
}

# Same cause, one step removed: `file_extension` is the leftmost dot-suffix of
# the *whole* URL that ends at `?`, `#` or the end of it, so a real query like
# `?url=beacon.gif` sets it and a filler query cannot. Every extension one-hot
# is therefore reachable, not just the ones observed to move.
INTERFACE_LOSSES.update({
    f"ext_{extension}": "the harness rebuilds the query out of filler, and "
                        "file_extension is read from the whole URL"
    for extension in ("js", "gif", "png", "jpg", "html", "php", "json", "css")
})

# `sql/05_per_request_full.sql`, restated. Deliberately a restatement and not
# an import of the crate's or the estimator's view of a URL: the point is to
# derive the log's columns from the URL independently and see the crate agree.
URL_PATH = re.compile(r"https?://[^/]+(/[^?#]*)")
FILE_EXTENSION = re.compile(r"\.([a-zA-Z0-9]+)(?:\?|#|$)")


@pytest.fixture(scope="session")
def extension():
    """The built `xgb_classifier` extension, or a skip."""
    return pytest.importorskip(
        "xgb_classifier",
        reason="xgb_classifier extension not built; "
               "run `maturin develop --release` in xgb-classifier/",
    )


def _derive(url: str) -> dict[str, object]:
    """The log columns a browser-facing estimator can rebuild from a URL."""
    path = URL_PATH.search(url)
    path = path.group(1) if path else None
    extension = FILE_EXTENSION.search(url)
    _, _, query = url.partition("?")
    return {
        "url_path": path,
        "path_depth": (path or "/").count("/"),
        "file_extension": extension.group(1).lower() if extension else None,
        "has_query_params": "?" in url,
        "url_length": len(url),
        # BigQuery's `SPLIT('', '&')` is one empty element, so a URL with no
        # query counts 1 and not 0.
        "num_query_params": 1 + query.count("&") if "?" in url else 1,
    }


@pytest.fixture(scope="session")
def comparison(split: Split) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """A sample of test rows, their rebuilt URLs, and the feature frame the
    crate is held to.

    The frame is what `engineer_features` produces from the inputs the
    interface can actually carry, which is now everything except the query: the
    URL columns are re-derived from the rebuilt URL rather than read from the
    log, and every other column is left exactly as the log has it.
    `INTERFACE_LOSSES` is the list of features the difference shows up in, and
    the tests below check it is complete.
    """
    for artifact in (solutions_mod.SHIPPED_MODEL, solutions_mod.SHIPPED_EMBEDDER):
        if not artifact.exists():
            pytest.skip(f"{artifact.relative_to(ROOT)} not present (gitignored)")

    sample = (split.test.sample(min(SAMPLE_ROWS, len(split.test)),
                                random_state=SAMPLE_SEED)
              .reset_index(drop=True))
    urls = solutions_mod.estimator_urls(sample)

    limited = sample.copy()
    for column, values in pd.DataFrame([_derive(url) for url in urls]).items():
        limited[column] = values.to_numpy()


    embedder = URLEmbedder(
        n_components=solutions_mod.N_EMBEDDING_COMPONENTS
    ).load(solutions_mod.SHIPPED_EMBEDDER)
    features = engineer_features(
        limited, solutions_mod.csv_training_split().train,
        embedder.transform(limited["url_path"].fillna("")))

    return sample, urls, features


def _call_args(extension, sample: pd.DataFrame, urls: list[str]):
    """`(url, context, initiator, method)` per row, as the harness passes them."""
    context_of = {resource: getattr(extension.RequestContext, name)
                  for resource, name in solutions_mod.REQUEST_CONTEXTS.items()}
    other = extension.RequestContext.OTHER
    initiator_of = {name.lower(): getattr(extension.RequestInitiator, name)
                    for name in ("PARSER", "SCRIPT", "OTHER")}
    unknown = extension.RequestInitiator.UNKNOWN
    return [
        (url, context_of.get(resource, other),
         initiator_of.get(initiator, unknown),
         method if isinstance(method, str) else None)
        for url, resource, initiator, method in zip(
            urls, sample["resource_type"].astype(str),
            sample["initiator_type"].astype("object").to_numpy(),
            sample["http_method"].astype("object").to_numpy())
    ]


def _rust_features(extension, sample: pd.DataFrame, urls: list[str]) -> np.ndarray:
    return np.array([extension.feature_vector(*args)
                     for args in _call_args(extension, sample, urls)],
                    dtype=np.float64)


def test_feature_vector_matches_engineer_features(extension, comparison) -> None:
    """Given the inputs the interface can carry, the port is `engineer_features`.

    Column by column rather than in aggregate: the ensemble is sensitive enough
    that a 1e-7 shift in one embedding moves the prediction, so a mismatch in
    the total says nothing about which of the 80 features is wrong.
    """
    sample, urls, expected = comparison
    got = _rust_features(extension, sample, urls)

    assert got.shape == expected.shape
    worst = np.abs(got - expected.to_numpy(dtype=np.float64)).max(axis=0)
    offenders = {name: float(gap)
                 for name, gap in zip(expected.columns, worst)
                 if gap > FEATURE_TOLERANCE}
    assert not offenders, (
        "xgb_classifier's feature port no longer reproduces engineer_features: "
        + ", ".join(f"{name} off by {gap:.3g}" for name, gap in offenders.items())
    )


def test_estimate_matches_the_shipped_artifact(extension, comparison) -> None:
    """`estimate_resources` is the artifact's own prediction, rounded.

    Feeding the crate's feature vector back through XGBoost isolates what this
    checks to the parts the crate does itself: the transpiled trees, the log
    link and intercept m2cgen does not apply, and the `+1` offset the Tweedie
    fit was trained with. The features are checked separately above.
    """
    sample, urls, _ = comparison
    features = _rust_features(extension, sample, urls)

    model = xgb.XGBRegressor()
    model.load_model(str(solutions_mod.SHIPPED_MODEL))
    want = np.clip(model.predict(features.astype(np.float32)) - 1, 0, None)

    got = np.array([extension.estimate_resources(*args)[0]
                    for args in _call_args(extension, sample, urls)],
                   dtype=np.float64)

    # Two differences are licensed: rounding to whole bytes, and m2cgen summing
    # the float32 leaves in float64 where XGBoost sums them in float32. The
    # second is relative, and lands at ~1e-5 of the estimate — six bytes on a
    # 600 KB asset — so the bar is a byte or 1e-4, whichever is larger.
    gap = np.abs(got - np.round(want))
    allowed = np.maximum(1.0, 1e-4 * want)
    assert np.all(gap <= allowed), (
        f"{int((gap > allowed).sum())} of {len(gap)} estimates differ from "
        f"model.predict by more than float32-vs-float64 accumulation explains; "
        f"worst {gap.max():.1f} bytes on an estimate of "
        f"{want[np.argmax(gap - allowed)]:.0f}"
    )
    # Almost all of them should be bit-identical, not merely close.
    assert (gap == 0.0).mean() > 0.95, (
        f"only {(gap == 0.0).mean():.1%} of estimates match model.predict "
        "exactly; the port has drifted from reproducing the artifact"
    )


def test_interface_loss_is_confined_to_known_features(extension, comparison,
                                                      dataset: str) -> None:
    """Nothing beyond `INTERFACE_LOSSES` differs from the faithful features.

    The other tests hold the crate to what the interface can carry. This one
    holds the *interface* to what it is documented to cost: it compares against
    `engineer_features` on the unmodified log row, and every column that then
    disagrees has to be named in `INTERFACE_LOSSES`. A feature quietly lost to
    a URL-parsing difference would show up here and nowhere else.
    """
    sample, urls, _ = comparison
    got = _rust_features(extension, sample, urls)

    embedder = URLEmbedder(
        n_components=solutions_mod.N_EMBEDDING_COMPONENTS
    ).load(solutions_mod.SHIPPED_EMBEDDER)
    faithful = engineer_features(
        sample, solutions_mod.csv_training_split().train,
        embedder.transform(sample["url_path"].fillna("")))

    worst = np.abs(got - faithful.to_numpy(dtype=np.float64)).max(axis=0)
    differing = {name for name, gap in zip(faithful.columns, worst)
                 if gap > FEATURE_TOLERANCE}

    unexpected = differing - set(INTERFACE_LOSSES)
    assert not unexpected, (
        f"{dataset}: {sorted(unexpected)} differ from the log's own features "
        "and are not in INTERFACE_LOSSES — either the port has a bug or the "
        "interface costs something new"
    )
    # The five features that used to dominate this list are reachable through
    # the interface now — four as parameters of `estimate_resources`, and
    # `rt_other` once `RequestContext` gained a JSON variant. Naming them here
    # is what stops them quietly regressing back into the tolerated set.
    reachable = {"init_script", "init_parser", "init_other", "is_post",
                 "rt_other", "domain_type_median"}
    assert not (differing & reachable), (
        f"{dataset}: {sorted(differing & reachable)} differ from the log's "
        "own, but the interface carries the initiator, the method and a JSON "
        "context — so they should be exact"
    )


def test_shipping_the_model_costs_orders_of_magnitude_more_than_the_table(
        extension) -> None:
    """The comparison the crate exists to make, as a number.

    `llm-classifier-python` is held to 50 KB over the baseline extension by
    `test_browsing_journey.test_llmclassifier_size`. The same model that scores
    worse than it on journey totals does not fit in that budget by three orders
    of magnitude, and most of the excess is not the trees: it is the
    truncated-SVD basis the 50 `url_emb_*` features are projected through,
    which cannot be quantised without moving the predictions (see
    `xgb-classifier/src/embedding.rs`).

    Measured over `llm_classifier_baseline`, the empty extension, for the same
    reason that test does: a PyO3 cdylib is ~430 KB before it holds anything,
    and what is being compared is what each estimator adds to one.

    Asserted as a floor rather than a ceiling. A ceiling would be a size budget
    this crate is not trying to meet, whereas the floor fails if the crate ever
    shrinks to something table-sized — which would mean it had stopped
    compiling in the whole model.
    """
    import llm_classifier

    def library(package: str) -> Path:
        # Located by path, not by import: `llm_classifier_baseline`'s
        # `#[pymodule]` is named `llm_classifier`, so importing it by its
        # package name fails. `test_browsing_journey.test_llmclassifier_size`
        # stats it for the same reason.
        directory = Path(llm_classifier.__file__).parent.parent / package
        libraries = sorted(directory.glob("*.abi3.so"))
        assert len(libraries) == 1, f"expected one extension in {directory}"
        return libraries[0]

    empty = library("llm_classifier_baseline").stat().st_size
    model = library("xgb_classifier").stat().st_size - empty
    table = library("llm_classifier").stat().st_size - empty
    print(f"\nover the baseline extension: xgb_classifier {model / 1e6:.2f} MB, "
          f"llm_classifier {table / 1024:.0f} KiB ({model / table:.0f}x)")

    assert model > 100 * table, (
        f"xgb_classifier adds only {model / table:.0f}x what the table does; "
        "it is probably no longer compiling in the whole model"
    )
