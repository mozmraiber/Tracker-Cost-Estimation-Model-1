"""Browsing-journey simulation over real HTTP Archive requests.

The per-request models exist to answer a product question: after a week of
browsing, how many bytes did Enhanced Tracking Protection save? That total is a
*sum over a correlated sample* of blocked requests, not an i.i.d. draw — a user
visits pages, and each page contributes whichever trackers it embeds. So the
metric that matters is the relative error of the journey total, which is what
this module measures.

Requests, URLs and sizes all come from `data/raw/per_request_1pct.csv`, kept
down to the hosts the Disconnect list matches — see `keep_disconnect_matched`.
Only the visit sequence is simulated: journeys draw whole pages from the crawl,
so the within-page correlation between tracker requests is preserved. That
correlation is what makes journey totals harder to predict than the uniform
per-request samples in `src/within_10pct_agg.py`.

The sampling and scoring here are dataset-agnostic — anything shaped like
`REQUIRED_COLUMNS` works. `http_archive` supplies a second such frame from the
parquet export, filtered by the `disconnect` DuckDB extension instead of by
`keep_disconnect_matched`; its docstring covers where the two populations
differ. Both carry real pages, so both are split and sampled the same way.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

import disconnect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "model"))

DEFAULT_REQUEST_LOG = ROOT / "data" / "raw" / "per_request_1pct.csv"

# Columns `train_multi_target.engineer_features` reads, plus the target and the
# page/domain keys the journey sampler and lookup tables need. Loading only
# these keeps a 320MB CSV manageable.
REQUIRED_COLUMNS = (
    "tracker_domain",
    "url_path",
    "path_depth",
    "file_extension",
    "has_query_params",
    "url_length",
    "num_query_params",
    "resource_type",
    "initiator_type",
    "http_method",
    "page_domain",
    "transfer_bytes",
)


# Per-request byte counts — either the recorded truth or a solution's
# predictions. One entry per row of the test frame.
Bytes = npt.NDArray[np.float64]

# One journey: the test-frame row positions contributed by its page visits.
# A page visited twice contributes its rows twice, as a real revisit would.
Journey = npt.NDArray[np.intp]
Journeys = list[Journey]

# Page domains, as `groupby` hands them back.
Pages = npt.NDArray[np.str_]


class JourneyStats(TypedDict):
    """What `journey_error` reports for one solution over many journeys.

    Percentages throughout, not fractions of one, except the `within_*` rates
    which are fractions of journeys.
    """

    n_journeys: int
    median_pct: float
    p90_pct: float
    within_10pct: float
    within_25pct: float
    signed_median_pct: float


def _page_bucket(pages: pd.Series, modulus: int) -> pd.Series:
    """Stable hash of a page domain into `modulus` buckets.

    Python's `hash` is salted per process, so it cannot be used for a subsample
    that has to be identical from run to run.
    """
    def digest(value: str) -> int:
        return int(hashlib.blake2b(str(value).encode(), digest_size=8).hexdigest(), 16)

    return pages.map(lambda v: digest(v) % modulus)


def tracker_hosts(hosts: Iterable[str]) -> set[str]:
    """The subset of `hosts` that `disconnect.is_tracker` calls a tracker.

    Computed once per distinct host, because matching is the expensive part and
    a host repeats across thousands of requests.

    `is_tracker` rather than `tracker_index`: the two disagree on the Content
    category, which lists a tracking company's own first-party properties, and
    a browser does not block those. Using the index would keep 225 such hosts
    here — fundingchoicesmessages.google.com and static.xx.fbcdn.net are the
    largest — and claim their bytes as savings.
    """
    return {host for host in dict.fromkeys(hosts)
            if disconnect.is_tracker(f"https://{host}/")}


def keep_disconnect_matched(requests: pd.DataFrame) -> pd.DataFrame:
    """Drop requests whose host the Disconnect list does not match.

    The savings figure only ever claims bytes for requests ETP actually
    blocked, and `disconnect.is_tracker` is what decides that — so those
    requests are the population a journey total is about. It is also the
    population the shipped estimators are fitted over: `llm_classifier` has no
    answer for a host outside the list beyond a context-only guess.

    The notable exclusion is www.googletagmanager.com, which carries 57% of
    the raw log's blocked bytes but is absent from the list. Leaving it in made
    every solution's journey total a prediction about a request ETP would not
    have blocked.
    """
    hosts = requests["tracker_domain"].astype(str)
    return requests[hosts.isin(tracker_hosts(hosts))].reset_index(drop=True)


def load_request_log(path: Path = DEFAULT_REQUEST_LOG,
                     page_modulus: int = 20) -> pd.DataFrame:
    """Load a deterministic subsample of the blocked-request log.

    Keeps one page domain in `page_modulus`. The subsample is taken by *page*
    rather than by row so that every request recorded for a kept page comes
    along with it — a journey that visits a page needs its whole request set,
    and a row-wise sample would silently shrink each page's byte total.

    Rows with a missing target are dropped and negative sizes are clipped, the
    same cleaning the training scripts do, and rows outside the Disconnect list
    are dropped by `keep_disconnect_matched`.
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=list(REQUIRED_COLUMNS),
                             chunksize=400_000, low_memory=False):
        chunks.append(chunk[_page_bucket(chunk["page_domain"], page_modulus) == 0])

    requests = pd.concat(chunks, ignore_index=True)
    requests = requests[requests["transfer_bytes"].notna()].copy()
    requests["transfer_bytes"] = requests["transfer_bytes"].clip(lower=0)
    return keep_disconnect_matched(requests)


@dataclass(frozen=True)
class Split:
    """A page-disjoint train/validation/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_page(df: pd.DataFrame, seed: int = 42,
                  train_frac: float = 0.70, val_frac: float = 0.15) -> Split:
    """Split on `page_domain` so no page spans two splits.

    The project's training scripts split on rows, which is right for measuring
    per-request accuracy. Journeys need whole pages, and a row split would leak
    a page's other requests into training — so the split is by page here.
    """
    pages = np.sort(df["page_domain"].unique())
    pages = pages[np.random.default_rng(seed).permutation(len(pages))]

    first = int(train_frac * len(pages))
    second = int((train_frac + val_frac) * len(pages))

    def take(selected: Pages) -> pd.DataFrame:
        return df[df["page_domain"].isin(set(selected))].reset_index(drop=True)

    return Split(train=take(pages[:first]),
                 val=take(pages[first:second]),
                 test=take(pages[second:]))


def zipf_page_weights(pages: Pages, seed: int = 11,
                      exponent: float = 1.1) -> npt.NDArray[np.float64]:
    """Assign Zipf visit probabilities to pages, in random rank order.

    The crawl records one visit per page, so it carries no popularity signal.
    Real browsing is heavily skewed — a few sites account for most visits — and
    that skew concentrates journeys onto a small set of pages, which is the
    harder case for a predictor. Used only by the popularity-weighted test;
    the budget tests draw pages uniformly and assume nothing.
    """
    weights = 1.0 / np.arange(1, len(pages) + 1) ** exponent
    ranks = np.random.default_rng(seed).permutation(len(pages))
    return np.asarray(weights[ranks] / weights.sum(), dtype=np.float64)


def sample_journeys(test_df: pd.DataFrame, pages_visited: int,
                    n_journeys: int, seed: int = 3,
                    popularity_weighted: bool = False) -> Journeys:
    """Draw `n_journeys` journeys of `pages_visited` page visits each.

    Returns one array of `test_df` row positions per journey — every tracker
    request recorded for each visited page. Pages are drawn with replacement,
    so a journey can revisit a site, as browsing does.
    """
    page_rows = test_df.groupby("page_domain", sort=True).indices
    pages = np.array(list(page_rows))

    probs = zipf_page_weights(pages) if popularity_weighted else None

    rng = np.random.default_rng(seed)
    journeys: Journeys = []
    for _ in range(n_journeys):
        chosen = rng.choice(len(pages), size=pages_visited, replace=True, p=probs)
        journeys.append(np.concatenate([page_rows[pages[i]] for i in chosen]))
    return journeys


def journey_error(y_true: Bytes, y_pred: Bytes,
                  journeys: Journeys) -> JourneyStats:
    """Relative error of the predicted journey total, across journeys.

    `signed_median_pct` is the calibration check: a predictor can have a small
    typical error while being systematically low, which would understate the
    savings figure every single week.
    """
    errors: list[float] = []
    signed: list[float] = []
    for rows in journeys:
        actual = float(y_true[rows].sum())
        if actual == 0:
            # No bytes blocked, so a relative error is undefined.
            continue
        predicted = float(y_pred[rows].sum())
        signed.append((predicted - actual) / actual * 100.0)
        errors.append(abs(predicted - actual) / actual * 100.0)

    if not errors:
        raise ValueError("every sampled journey blocked zero bytes")

    magnitudes = np.asarray(errors)
    return {
        "n_journeys": len(errors),
        "median_pct": float(np.median(magnitudes)),
        "p90_pct": float(np.percentile(magnitudes, 90)),
        "within_10pct": float(np.mean(magnitudes <= 10.0)),
        "within_25pct": float(np.mean(magnitudes <= 25.0)),
        "signed_median_pct": float(np.median(signed)),
    }
