"""What does `estimate_resources(url, context)` cost the shipped model?

`xgb_classifier` compiles in the same artifact as `tests/solutions.py`'s
`xgboost_shipped`, and `tests/test_xgb_classifier_port.py` holds its 80
features to the bit against `engineer_features`'. It still scores worse on
journey totals, because the interface cannot carry everything the model was
fitted on. This script attributes that gap, one input at a time:

    python xgb-classifier/scripts/interface_ablation.py
    python xgb-classifier/scripts/interface_ablation.py --dataset http_archive

Each row degrades one more input of the Python predictor and re-scores it. Only
one loss is left, and it is this harness's rather than the interface's: the log
kept each URL's length and parameter count but not the query, so
`tests/solutions.estimator_urls` rebuilds one matching both out of filler. That
is exact for `num_query_params` and not for `file_extension`, which the SQL
reads off the whole URL rather than the path, so a real `?url=beacon.gif` sets
it and filler cannot. A browser holds the real URL and would lose nothing.

The crate should land on that row exactly, and does.

The rows after it price what the interface gained, by taking it back one piece
at a time: first `estimate_resources`' `initiator` and `method` parameters,
then `RequestContext.JSON`. Between them they were the entire gap the crate
used to give up — 23 points on the CSV extract.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
for path in (ROOT / "tests", ROOT / "src" / "model"):
    sys.path.insert(0, str(path))

import conftest  # noqa: E402
import http_archive  # noqa: E402
import journey  # noqa: E402
import solutions  # noqa: E402
from train_multi_target import engineer_features  # noqa: E402
from url_embeddings import URLEmbedder  # noqa: E402

# `sql/05_per_request_full.sql`'s `file_extension`, which has to be re-derived
# once the query is no longer the one the log measured.
FILE_EXTENSION = re.compile(r"\.([a-zA-Z0-9]+)(?:\?|#|$)")


def load_split(dataset: str) -> journey.Split:
    """`conftest`'s split for `dataset`, so the scores here are its scores."""
    if dataset == "csv":
        log = journey.load_request_log(journey.DEFAULT_REQUEST_LOG,
                                       page_modulus=conftest.PAGE_MODULUS)
    else:
        unavailable = http_archive.available(dataset)
        if unavailable:
            raise SystemExit(unavailable)
        log = http_archive.load_request_log(dataset)
    return journey.split_by_page(log, seed=42)


def stages(test: pd.DataFrame, urls: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """The log frame, degraded one input at a time.

    The walk down to the crate is one step long now. The two entries after it
    go further, taking back the initiator parameters and then the JSON context,
    to price what the interface gained by carrying them.
    """
    extensions = [match.group(1).lower() if match else None
                  for match in (FILE_EXTENSION.search(url) for url in urls)]

    frame = test.copy()
    out = [("xgboost_shipped (the log's own columns)", frame.copy())]

    frame["num_query_params"] = [
        1 + url.split("?", 1)[1].count("&") if "?" in url else 1 for url in urls]
    frame["url_length"] = [len(url) for url in urls]
    frame["has_query_params"] = ["?" in url for url in urls]
    frame["file_extension"] = extensions
    out.append(("query rebuilt out of filler = the crate", frame.copy()))

    frame = frame.copy()
    frame["initiator_type"] = np.nan
    frame["http_method"] = np.nan
    out.append(("...and without the initiator parameters", frame))

    frame = frame.copy()
    frame["resource_type"] = frame["resource_type"].replace({"json": "other"})
    out.append(("...and without a JSON context", frame))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="csv", choices=conftest.DATASETS)
    parser.add_argument("--seed", type=int, default=conftest.JOURNEY_SEED,
                        help="journey seed; the suite's budgets take the worst "
                             "of 3, 5 and 7")
    args = parser.parse_args()

    for artifact in (solutions.SHIPPED_MODEL, solutions.SHIPPED_EMBEDDER):
        if not artifact.exists():
            raise SystemExit(f"{artifact.relative_to(ROOT)} is missing (gitignored)")

    split = load_split(args.dataset)
    test = split.test
    actual = test["transfer_bytes"].clip(lower=0).to_numpy(dtype=float)
    urls = solutions.estimator_urls(test)
    journeys = journey.sample_journeys(
        test, pages_visited=conftest.PAGES_VISITED,
        n_journeys=conftest.N_JOURNEYS, seed=args.seed)

    embedder = URLEmbedder(
        n_components=solutions.N_EMBEDDING_COMPONENTS
    ).load(solutions.SHIPPED_EMBEDDER)
    model = xgb.XGBRegressor()
    model.load_model(str(solutions.SHIPPED_MODEL))
    # The encoding frame is the log the artifact was fitted on, whatever it is
    # being scored against -- see `solutions.csv_training_split`.
    encoded_from = solutions.csv_training_split().train

    def score(preds: np.ndarray) -> tuple[float, float]:
        stats = journey.journey_error(actual, preds, journeys)
        return stats["median_pct"], stats["signed_median_pct"]

    print(f"{args.dataset}: {len(test)} test rows, "
          f"{(test['resource_type'] == 'json').mean():.1%} json, "
          f"{(test['num_query_params'] > 1).mean():.1%} multi-parameter, "
          f"{test['initiator_type'].notna().mean():.1%} with an initiator type")
    print(f"journey seed {args.seed}, {conftest.N_JOURNEYS} journeys of "
          f"{conftest.PAGES_VISITED} pages\n")
    print(f"  {'predictor':44s} {'median':>8s} {'signed':>9s}")

    for label, frame in stages(test, urls):
        features = engineer_features(
            frame, encoded_from, embedder.transform(frame["url_path"].fillna("")))
        preds = np.clip(model.predict(features) - 1, 0, None).astype(np.float64)
        median, signed = score(preds)
        print(f"  {label:44s} {median:7.2f}% {signed:+8.2f}%")

    median, signed = score(solutions.xgb_classifier_m2cgen(split))
    print(f"\n  {'xgb_classifier (the crate, measured)':44s} {median:7.2f}% "
          f"{signed:+8.2f}%")


if __name__ == "__main__":
    main()
