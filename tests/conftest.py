"""Fixtures for the browsing-journey error tests.

Every journey test runs once per dataset in `DATASETS`: the BigQuery CSV
extract the project was developed against, and the HTTP Archive parquet export
that `http_archive` narrows with the `disconnect` DuckDB extension. They are
not two samples of one population — see `http_archive`'s docstring — so budgets
are per dataset.

The expensive work — loading the request log, fitting URL embeddings and
training the model — happens once per dataset and is shared by every test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# `journey` and `solutions` live here; the model code they reuse lives in src/model.
for _path in (str(Path(__file__).resolve().parent), str(ROOT / "src" / "model")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import http_archive as http_archive_mod  # noqa: E402
import journey as journey_mod  # noqa: E402
import solutions as solutions_mod  # noqa: E402
from journey import Bytes, Journeys, Split  # noqa: E402


class PredictFn(Protocol):
    """Resolves a solution name to its predictions over `split.test`."""

    def __call__(self, name: str) -> Bytes: ...


# Journey shape. 40 page visits is roughly a day of browsing and yields a few
# hundred blocked tracker requests, the regime the paper's aggregation analysis
# covers. 600 journeys keeps the median stable without dominating the runtime.
PAGES_VISITED = 40
N_JOURNEYS = 600
JOURNEY_SEED = 3

# Keep 1 page domain in 20, which is ~89k in-scope requests over ~41k pages —
# enough for the lookup tables to have coverage while the suite still runs in
# seconds. `http_archive.Export` carries a `page_modulus` picked to land the
# export at about the same row count, so the datasets are comparable.
PAGE_MODULUS = 20

DATASETS = ("csv", "http_archive")


@pytest.fixture(scope="session", params=DATASETS)
def dataset(request: pytest.FixtureRequest) -> str:
    """Which request log the journey tests are running against."""
    return request.param


@pytest.fixture(scope="session")
def request_log(dataset: str) -> pd.DataFrame:
    """Deterministic subsample of the blocked-request log."""
    if dataset == "csv":
        path = journey_mod.DEFAULT_REQUEST_LOG
        if not path.exists():
            pytest.skip(
                f"{path.relative_to(ROOT)} not present (gitignored); regenerate it "
                f"with the BigQuery extract in sql/ to run the journey tests"
            )
        return journey_mod.load_request_log(path, page_modulus=PAGE_MODULUS)

    unavailable = http_archive_mod.available(dataset)
    if unavailable:
        pytest.skip(unavailable)
    return http_archive_mod.load_request_log(dataset)


@pytest.fixture(scope="session")
def split(request_log: pd.DataFrame) -> Split:
    """Page-disjoint train/val/test split of the sampled log.

    `split_by_page`'s default 70/15 share for both datasets. A per-dataset
    absolute page budget used to live here, for a 50%-sampled export large
    enough that letting its training half grow with its evaluation half would
    have stopped the datasets being comparable. That export has been removed.
    """
    return journey_mod.split_by_page(request_log, seed=42)


@pytest.fixture(scope="session")
def journeys(split: Split) -> Journeys:
    return journey_mod.sample_journeys(
        split.test, pages_visited=PAGES_VISITED,
        n_journeys=N_JOURNEYS, seed=JOURNEY_SEED,
    )


@pytest.fixture(scope="session")
def predict(split: Split) -> PredictFn:
    """Look up a solution's test-set predictions, building it on first use.

    Lazy and memoized: `xgboost_tweedie` is only trained if a test asks for it,
    and only once. A solution whose artifacts are missing skips the tests that
    need it instead of failing the session.
    """
    cache: dict[str, Bytes | FileNotFoundError] = {}

    def get(name: str) -> Bytes:
        if name not in cache:
            try:
                cache[name] = solutions_mod.ALL_SOLUTIONS[name](split)
            except FileNotFoundError as exc:
                cache[name] = exc
        result = cache[name]
        if isinstance(result, FileNotFoundError):
            pytest.skip(f"{name}: missing artifact {Path(str(result)).name} (gitignored)")
        return result

    return get
