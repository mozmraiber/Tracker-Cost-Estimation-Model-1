"""The candidate predictors compared in the journey-error tests.

Each builder takes the split and returns predicted
`transfer_bytes` for every row of the test frame. The lookup tables, the URL
embeddings and the feature matrix all come from `src/model`, so these tests
exercise the shipped implementations rather than a restatement of them.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from journey import (ROOT, Bytes, Split, load_request_log, split_by_page)
from train_multi_target import engineer_features, lut_baseline_fast
from url_embeddings import URLEmbedder

# A solution: fit whatever it needs on the split's train half, then predict
# `transfer_bytes` for every row of `split.test`.
Predictor = Callable[[Split], Bytes]

# Which way a deliberately broken predictor is expected to be wrong.
BiasDirection = Literal["low", "high"]


@dataclass(frozen=True)
class Budget:
    """How accurate a solution's journey totals have to be.

    Two numbers, because one does not pin a heavy-tailed error down: a
    predictor can hold a decent median while a tenth of journeys are wildly
    off, and the dashboard shows every journey.

    `median_pct` caps the typical relative error of a journey total;
    `within_25pct` floors the fraction of journeys landing within a quarter of
    the truth. Both are set off the measured value with headroom — across
    journey seeds the median moves by 1-2 points and the rate by ~0.04 — so a
    failure means the solution regressed, not that the sample shifted.
    """

    median_pct: float
    within_25pct: float


TARGET = "transfer_bytes"

# Laplace weight for the smoothed path table, the value `smoothed_lut.py`
# selects on this data.
SMOOTHING_K = 5.0

N_EMBEDDING_COMPONENTS = 50

# The `RequestContext` variant each of the log's `resource_type` values maps
# to. A value whose variant the extension does not define falls to OTHER, which
# is what `_rust_extension` does with it — so this is read against whichever
# extension is being called, not against one of them.
#
# That matters for exactly one entry. `llm_classifier` has no JSON variant,
# because `build_table.py` fitted its table with `json` binned into OTHER (see
# `RESOURCE_CONTEXT` there, which is the source of truth for that crate), so it
# could not tell the two apart even if it were told. `xgb_classifier` does have
# one, and reads a different row of its target encodings for it.
REQUEST_CONTEXTS = {
    "script": "SCRIPT", "image": "IMAGE", "video": "VIDEO", "other": "OTHER",
    "audio": "AUDIO", "css": "CSS", "font": "FONT", "html": "HTML",
    "text": "TEXT", "wasm": "WASM", "xml": "XML", "json": "JSON",
}

# Length of the `https://` that `url_length` counts but a domain and path do
# not, so the recorded query length can be recovered from the two.
SCHEME_LENGTH = len("https://")

MODEL_DIR = ROOT / "models" / "per_request"
SHIPPED_MODEL = MODEL_DIR / "xgb_transfer_bytes.json"
SHIPPED_EMBEDDER = MODEL_DIR / "url_embedder.joblib"


@functools.lru_cache(maxsize=1)
def csv_training_split() -> Split:
    """The CSV extract's split, which is the only thing XGBoost is fitted on.

    Both XGBoost solutions train on `data/raw/per_request_1pct.csv` whatever
    they are being evaluated against, so their scores on the parquet exports
    are transfer results and are comparable with each other. `xgboost_shipped`
    has no choice — its stored artifact was trained there — and
    `xgboost_tweedie` is held to the same rule so that the recipe and the
    artifact differ only in being refitted, not in what they saw.

    The lookup tables are *not* held to this: they are per-split baselines by
    definition, and the point of comparing against them is what a table fitted
    on the same data as the model would manage.

    Cached, because every dataset's session asks for the same split, and it
    matches what `conftest` builds for the `csv` dataset — same page modulus,
    same seed — so evaluating on the CSV is unchanged by this.

    Raises `FileNotFoundError` if the extract is absent; the `predict` fixture
    turns that into a skip.
    """
    if not ROOT.joinpath("data", "raw", "per_request_1pct.csv").exists():
        raise FileNotFoundError(ROOT / "data" / "raw" / "per_request_1pct.csv")
    return split_by_page(load_request_log(), seed=42)


def _target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].clip(lower=0)


def global_median(split: Split) -> Bytes:
    """Predict one constant for every request. Degenerate control."""
    return np.full(len(split.test), _target(split.train).median(),
                   dtype=np.float64)


def domain_median_lut(split: Split) -> Bytes:
    """Median bytes per tracker domain — the naive dashboard lookup."""
    train, test = split.train, split.test
    per_domain = _target(train).groupby(train["tracker_domain"]).median()
    return (test["tracker_domain"].map(per_domain)
            .fillna(_target(train).median())
            .to_numpy(dtype=np.float64))


def domain_type_median_lut(split: Split) -> Bytes:
    """Median per (domain, resource_type) — the paper's LUT baseline."""
    return np.asarray(lut_baseline_fast(split.train, split.test, TARGET),
                      dtype=np.float64)


def _domain_type_agg(split: Split, how: str) -> Bytes:
    """(domain, resource_type) aggregate with domain then global fallback."""
    train, test = split.train, split.test
    keys = ["tracker_domain", "resource_type"]
    table = train.groupby(keys)[TARGET].agg(how).rename("_pred").reset_index()
    preds = test[keys].merge(table, on=keys, how="left")["_pred"]
    preds = preds.fillna(test["tracker_domain"].map(
        train.groupby("tracker_domain")[TARGET].agg(how)))
    return preds.fillna(train[TARGET].agg(how)).to_numpy(dtype=np.float64)


def domain_type_mean_lut(split: Split) -> Bytes:
    """Mean per (domain, resource_type).

    The mean is the right statistic for a total: summing conditional medians
    of a distribution that is ~44% zeros underestimates a sum badly, summing
    conditional means does not.
    """
    return _domain_type_agg(split, "mean")


def smoothed_path_lut(split: Split) -> Bytes:
    """Laplace-smoothed path-level table, per `src/model/smoothed_lut.py`.

    Shrinks each (domain, path) mean toward the global mean in proportion to
    how little evidence supports it, and falls back to the (domain, type) mean
    for paths never seen in training.
    """
    train, test = split.train, split.test
    keys = ["tracker_domain", "url_path"]
    stats = train.groupby(keys)[TARGET].agg(["mean", "count"]).reset_index()
    global_mean = train[TARGET].mean()

    merged = test[keys].merge(stats, on=keys, how="left")
    smoothed = ((merged["count"] * merged["mean"] + SMOOTHING_K * global_mean)
                / (merged["count"] + SMOOTHING_K))
    return np.where(merged["mean"].notna(), smoothed,
                    _domain_type_agg(split, "mean")).astype(np.float64)


def xgboost_tweedie(split: Split) -> Bytes:
    """XGBoost with Tweedie loss, the configuration `train_multi_target` ships.

    Trained here rather than loaded, so this covers the whole recipe — URL
    embeddings, feature engineering and loss choice — rather than a stored
    artifact. Tweedie handles the zero inflation directly, which is what gives
    the model its calibrated-total behaviour. The +1 offset mirrors the
    training script: Tweedie needs strictly positive targets.

    Fitted on `csv_training_split`, not on `split.train`, so what is measured
    on a parquet export is the recipe transferring to it.
    """
    fitted_on = csv_training_split()
    train, val, test = fitted_on.train, fitted_on.val, split.test

    embedder = URLEmbedder(n_components=N_EMBEDDING_COMPONENTS)
    embedder.fit(train["url_path"].fillna(""))

    def features(frame: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(
            frame, train, embedder.transform(frame["url_path"].fillna("")))

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        tree_method="hist",
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(features(train), _target(train) + 1,
              eval_set=[(features(val), _target(val) + 1)], verbose=False)
    return np.asarray(np.clip(model.predict(features(test)) - 1, 0, None),
                      dtype=np.float64)


def xgboost_shipped(split: Split) -> Bytes:
    """The trained artifact in `models/per_request`, loaded not retrained.

    This is the predictor the dashboard would actually ship, so it is checked
    as-is. Raises `FileNotFoundError` if the artifacts are absent; the fixture
    turns that into a skip.

    `engineer_features` target-encodes the tracker domain from a training
    frame, and that frame is `csv_training_split`'s — the log the artifact was
    trained on. Passing the evaluation dataset's own train half instead would
    hand a CSV-trained model an encoding fitted on data it never saw, which is
    both wrong and a way of training on the eval set.
    """
    for artifact in (SHIPPED_MODEL, SHIPPED_EMBEDDER):
        if not artifact.exists():
            raise FileNotFoundError(artifact)

    embedder = URLEmbedder(n_components=N_EMBEDDING_COMPONENTS).load(SHIPPED_EMBEDDER)
    embeddings = embedder.transform(split.test["url_path"].fillna(""))
    features = engineer_features(split.test, csv_training_split().train, embeddings)

    model = xgb.XGBRegressor()
    model.load_model(str(SHIPPED_MODEL))
    expected = model.get_booster().feature_names
    if list(features.columns) != expected:
        raise AssertionError(
            "feature matrix no longer matches the shipped model; regenerate "
            f"{SHIPPED_MODEL.relative_to(ROOT)}")

    return np.asarray(np.clip(model.predict(features) - 1, 0, None),
                      dtype=np.float64)


def _filler_query(length: int, params: int) -> str:
    """A query `length` characters long, counting the `?`, that splits into
    `params` `&`-separated parts.

    The log recorded both numbers but not the query itself, so this is the most
    a rebuilt URL can carry. The byte budget goes on `params - 1` separators
    first and filler after, spread evenly — evenly only so it reads like a real
    query, since `SPLIT(q, '&')` counts one more than the number of `&`
    wherever they sit.

    A query too short to hold its own separators keeps as many as fit, which is
    the one case where `num_query_params` still comes out low.
    """
    if length <= 0:
        return ""
    separators = min(max(params - 1, 0), length - 1)
    filler = length - 1 - separators
    parts = [filler // (separators + 1)] * (separators + 1)
    for i in range(filler % (separators + 1)):
        parts[i] += 1
    return "?" + "&".join("q" * part for part in parts)


def estimator_urls(test: pd.DataFrame) -> list[str]:
    """Rebuild each row's request URL, as an in-browser estimator would see it.

    Neither Rust extension takes a URL's pieces, only the URL, so the log's
    columns have to go back together into the string they came from.
    The log kept each URL's length and its parameter count but not the query,
    so the query is rebuilt out of filler to match both — see `_filler_query`.

    Matching the count as well as the length matters to `xgb_classifier`, whose
    model has a `num_query_params` feature; it makes no difference to
    `llm_classifier`, which reads nothing from a query but its log2 length, and
    the length is unchanged. What no reconstruction can recover is
    `file_extension`, which `sql/05_per_request_full.sql` reads off the whole
    URL, so a real `?url=beacon.gif` sets it and filler cannot. A browser holds
    the real URL and has neither problem: that residue belongs to the harness,
    not to the estimator.
    """
    domains = test["tracker_domain"].astype(str)
    paths = test["url_path"].fillna("/").astype(str)
    query_lengths = np.clip(
        test["url_length"].fillna(0).to_numpy(dtype=np.int64)
        - domains.str.len().to_numpy() - paths.str.len().to_numpy()
        - SCHEME_LENGTH, 0, None)
    # The log's count is 1 for a URL with no query at all, because BigQuery
    # splits the empty string into one empty element; `_filler_query` returns
    # nothing for those, so the 1 never reaches it.
    param_counts = test["num_query_params"].fillna(1).to_numpy(dtype=np.int64)

    return [f"https://{domain}{path}" + _filler_query(length, params)
            for domain, path, length, params
            in zip(domains.to_numpy(), paths.to_numpy(), query_lengths,
                   param_counts)]


def _rust_extension(module_name: str, split: Split) -> Bytes:
    """Predictions from one of the Rust extensions, over `split.test`.

    Both expose `estimate_resources(url, context, initiator, method) ->
    (bytes, cpu_ms)` and the same `RequestContext` and `RequestInitiator`, so
    one caller serves both and the two are measured through identical inputs.
    They read the arguments to different ends — `xgb_classifier` one-hots the
    initiator and a POST flag into its feature vector, `llm_classifier` keys
    only on whether the method forbids a response body — but each gets the
    same four values for every row.

    `xgb_classifier` still accepts one thing more, a JSON request context of
    its own, which is used when the extension defines it. With everything
    supplied it scores what `xgboost_shipped` scores; the two budgets are equal
    for that reason.

    Only the byte half is scored: `cpu_ms` is derived from it by coefficients
    neither crate fits, and the log has no CPU column to score it against.

    Raises `FileNotFoundError` if the extension is not built; the `predict`
    fixture turns that into a skip.
    """
    try:
        extension = importlib.import_module(module_name)
    except ImportError as exc:
        raise FileNotFoundError(f"{exc.name} extension (build it with maturin)") from exc

    context_of = {resource: getattr(extension.RequestContext, name)
                  for resource, name in REQUEST_CONTEXTS.items()
                  if hasattr(extension.RequestContext, name)}
    other = extension.RequestContext.OTHER

    urls = estimator_urls(split.test)
    resources = split.test["resource_type"].astype(str).to_numpy()

    initiators = extension.RequestInitiator
    initiator_of = {name.lower(): getattr(initiators, name)
                    for name in ("PARSER", "SCRIPT", "OTHER")}
    # The log's `preflight`, `FedCM` and `preload` have no variant in either
    # extension, and neither does a row that recorded nothing.
    unknown = initiators.UNKNOWN
    # A row with no method recorded reaches the extensions as the empty string,
    # which is how both spell "not known": it is not POST for `xgb_classifier`
    # and not a bodyless method for `llm_classifier`, so each answers as it did
    # before either took a method at all.
    extra = [
        (initiator_of.get(initiator, unknown),
         method if isinstance(method, str) else "")
        for initiator, method in zip(
            split.test["initiator_type"].astype("object").to_numpy(),
            split.test["http_method"].astype("object").to_numpy())
    ]

    preds = np.empty(len(urls), dtype=np.float64)
    for i, (url, resource) in enumerate(zip(urls, resources)):
        preds[i], _cpu_ms = extension.estimate_resources(
            url, context_of.get(resource, other), *extra[i])
    return preds


def xgb_classifier_m2cgen(split: Split) -> Bytes:
    """The Rust `xgb_classifier` extension: `xgboost_shipped`, compiled in.

    The same artifact as `xgboost_shipped` — `xgb_transfer_bytes.json`, its 370
    trees transpiled to Rust by m2cgen — behind `llm_classifier_table`'s
    interface, so the model and the table can be priced against each other on
    what it costs a browser to ship them. `xgb-classifier/scripts/build_model.py`
    builds it and `xgb-classifier/src/lib.rs` records the comparison.

    It is not a restatement of the model: `tests/test_xgb_classifier_port.py`
    holds its 80 features to within 2 ulp of `engineer_features`'. So its score
    here should track `xgboost_shipped`'s, and where it does not, the gap is
    the interface — a URL and a context cannot carry the initiator type or the
    HTTP method, and this harness's rebuilt query cannot carry
    `num_query_params`. See `estimator_urls` and `features::build` in the
    crate.

    Raises `FileNotFoundError` if the extension is not built; the fixture turns
    that into a skip.
    """
    return _rust_extension("xgb_classifier", split)


def llm_classifier_table(split: Split) -> Bytes:
    """The Rust `llm_classifier` extension, called the way Firefox calls it.

    A hierarchy of conditional means over URL features, fitted offline by
    `llm-classifier/scripts/build_table.py` and compiled into the extension.
    Unlike every other solution here it is not fitted on `split.train`: its
    table is built over all 4.5M in-scope requests of the HTTP Archive export
    (`build_table.py --parquet`), test pages included. So its `http_archive`
    score is an in-sample best case. Its `csv` score is drawn from a different
    export, but not a clean holdout either — both are HTTP Archive crawls and
    the popular asset URLs recur, so the fit has already seen the exact path
    of 93% of the CSV's test rows. Read both as in-sample, which is the regime
    the estimator ships in: the table is compiled in, not learned on device.
    It is measured because it is the estimator that would ship in the browser,
    and a journey total it got wrong would be wrong on the dashboard.

    Raises `FileNotFoundError` if the extension is not built; the fixture
    turns that into a skip.
    """
    return _rust_extension("llm_classifier", split)


def double_counted_mean_lut(split: Split) -> Bytes:
    """The mean lookup table, counted twice. Deliberately wrong.

    Stands in for a plausible real bug — attributing each blocked request to
    the total twice — so the journey metric is shown to catch a bias it should
    catch, in the opposite direction from `global_median`.
    """
    return 2.0 * domain_type_mean_lut(split)


# Predictors put forward as usable solutions, with the journey-total accuracy
# each is expected to hold. The ladder is deliberately visible here: a naive
# per-domain lookup is allowed to be five times as wrong as the shipped model,
# because the point of the ladder is that the error falls as the estimator
# learns more of the URL.
SOLUTIONS: dict[str, tuple[Predictor, Budget]] = {
    "domain_median_lut": (domain_median_lut, Budget(55.0, 0.22)),
    "domain_type_median_lut": (domain_type_median_lut, Budget(30.0, 0.50)),
    "domain_type_mean_lut": (domain_type_mean_lut, Budget(30.0, 0.50)),
    "smoothed_path_lut": (smoothed_path_lut, Budget(26.0, 0.55)),
    # Re-measured when `keep_disconnect_matched` switched to `is_tracker`:
    # dropping the Content-only hosts left a population Tweedie fits worse,
    # and it now runs ~22% low. See `KNOWN_UNCALIBRATED`.
    "xgboost_tweedie": (xgboost_tweedie, Budget(28.0, 0.58)),
    "xgboost_shipped": (xgboost_shipped, Budget(15.0, 0.70)),
    # The same artifact behind the shipping interface, and now the same
    # predictor: it scores what `xgboost_shipped` scores, to two decimal
    # places, so it is held to the same budget. It was 30.2% before
    # `estimate_resources` took the initiator and the method and
    # `RequestContext` gained a JSON variant; those were the whole gap.
    # `xgb-classifier/scripts/interface_ablation.py` prices each.
    "xgb_classifier_m2cgen": (xgb_classifier_m2cgen, Budget(15.0, 0.70)),
    "llm_classifier_table": (llm_classifier_table, Budget(10.0, 0.80)),
}

# Predictors that must *fail* the journey test, one biased low and one high.
# They are the negative control: if a bug ever made the simulation report
# small errors regardless of prediction quality, these would stop failing and
# give it away. The floor is the minimum journey error each must show.
BROKEN_SOLUTIONS: dict[str, tuple[Predictor, float, BiasDirection]] = {
    "global_median": (global_median, 50.0, "low"),
    "double_counted_mean_lut": (double_counted_mean_lut, 50.0, "high"),
}

# The same budgets re-measured on the HTTP Archive export. Both datasets cover
# the same population now that `is_tracker` leaves out the Content category —
# 100.8k requests over 41.9k pages against the CSV's 88.8k over 41.3k, with a
# similar resource-type mix and zero rate (45% against 51%). What still differs
# is how they were narrowed: the CSV by the Almanac's third-party host
# categories at export time, this one by `is_tracker` alone, so it carries
# hosts the Almanac list never had. That is worth a separate set of numbers,
# but no longer a wildly different one.
#
# Set the same way as the CSV budgets — the worst of journey seeds 3, 5 and 7
# with headroom. `xgboost_tweedie`'s median budget covers its worst case
# across every journey shape the suite tries, which is a 160-page journey
# (26.3%) rather than the plain 40-page draw the main test uses.
HTTP_ARCHIVE_BUDGETS: dict[str, Budget] = {
    "domain_median_lut": Budget(44.0, 0.24),
    "domain_type_median_lut": Budget(22.0, 0.60),
    "domain_type_mean_lut": Budget(19.0, 0.65),
    "smoothed_path_lut": Budget(15.0, 0.74),
    # Both XGBoost entries are CSV-trained transfers — see `csv_training_split`
    # — so what is measured here is the recipe carrying over, and it carries
    # over badly. Both improved by ~11-15 points when the export gained
    # `initiator_type` and `http_method`, which until then it selected as NULL.
    "xgboost_tweedie": Budget(47.0, 0.08),
    "xgboost_shipped": Budget(36.0, 0.35),
    # Identical to `xgboost_shipped` here, hence the identical budget: the
    # interface now carries every input the model was fitted on, and the one
    # thing it cannot — the query the log did not keep — costs nothing
    # measurable.
    "xgb_classifier_m2cgen": Budget(36.0, 0.35),
    # `MODEL`, so this has to cover its worst journey shape rather than just
    # the 40-page draw — which here is the popularity-weighted one at 9.4%,
    # not any of the lengths.
    "llm_classifier_table": Budget(11.0, 0.88),
}


DATASET_BUDGETS: dict[str, dict[str, Budget]] = {
    "csv": {},  # `SOLUTIONS` already carries the CSV budgets.
    "http_archive": HTTP_ARCHIVE_BUDGETS,
}


def budget_for(dataset: str, name: str) -> Budget:
    """The accuracy `name` has to hold on `dataset`."""
    return DATASET_BUDGETS[dataset].get(name, SOLUTIONS[name][1])


# Where a mean-calibrated estimator is biased anyway, and by how much. These
# are findings, not thresholds to relax: an unbiased total is the reason the
# mean-based estimators exist, so rather than widen `MAX_SIGNED_BIAS_PCT` for
# a dataset that breaks it, the specific pairs are recorded here.
# `test_totals_are_calibrated_not_merely_close` lets only the names in its
# `MEAN_CALIBRATED_SKIPPED` off with a skip; any other miss is a hard failure.
#
# Both XGBoost entries train on the CSV extract only, so across both datasets
# the model is fixed and only the evaluation set changes. That makes
# the shape of the problem legible: the recipe is ~22% low on the log it was
# fitted on and ~43-47% low on either parquet export, predicting about half
# their summed bytes. It is a transfer failure on top of a calibration one,
# and the two XGBoost solutions now agree with each other to within 2 points
# everywhere, as they should when they differ only in being refitted.
#
# Worth noting what is *not* here: `domain_type_mean_lut` and
# `smoothed_path_lut` stay calibrated on both, within +-5%, and so does
# `llm_classifier_table`. The plain conditional mean is
# doing the job the model was meant to do better.
KNOWN_UNCALIBRATED: dict[tuple[str, str], str] = {
    ("csv", "xgboost_tweedie"):
        "Tweedie runs ~22% low on journey totals, predicting 0.76x the test "
        "set's summed bytes; it was inside the bar until is_tracker() stopped "
        "counting the Content category, whose large static assets the fit "
        "leaned on",
    ("http_archive", "xgboost_tweedie"):
        "the CSV-fitted recipe transfers badly to this export: ~41% low, "
        "predicting 0.55x its summed bytes",
    ("http_archive", "xgboost_shipped"):
        "the shipped artifact transfers better than the refitted recipe but "
        "still badly, ~30% low, predicting 0.66x its summed bytes; it was "
        "~41% until the export gained initiator_type and http_method",
    # The same artifact as `xgboost_shipped`, so it inherits that bias and
    # adds what the interface still cannot carry. There is no `csv` entry any
    # more: once `estimate_resources` took the initiator and the method it came
    # inside the bar there, at -9.8%.
    ("http_archive", "xgb_classifier_m2cgen"):
        "~30% low, which is `xgboost_shipped`'s bias exactly — the same "
        "artifact, and now the same inputs, so the same transfer failure and "
        "nothing added by the interface",
}


ALL_SOLUTIONS: dict[str, Predictor] = {
    **{name: build for name, (build, _) in SOLUTIONS.items()},
    **{name: build for name, (build, _, _) in BROKEN_SOLUTIONS.items()},
}
