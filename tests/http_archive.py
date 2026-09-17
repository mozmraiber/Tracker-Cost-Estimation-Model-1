"""Journey inputs from the HTTP Archive parquet exports, filtered by DuckDB.

`journey.load_request_log` reads a BigQuery CSV that `sql/06_large_scale.sql`
had already narrowed, server-side, to the HTTP Archive Almanac's ad, analytics,
social, tag-manager and consent-provider hosts. This module covers the exports
in `data/`, which arrive unfiltered — the shape a full crawl actually comes in.
Narrowing them is the job of the `disconnect` DuckDB extension in
`duckdb-disconnect/`, whose `is_tracker` runs fast enough to sit in the `WHERE`
clause over all of it.

One export is registered. `http-archive-urls-1pct` samples **1% of requests**,
so a page keeps only two or three of the ~280 requests it actually made: plenty
of pages, but each a fragment, which is also true of the CSV extract. It is
subsampled by page and never by row, for the reason
`journey.load_request_log` gives — a journey that visits a page needs that
page's whole request set.

A 50%-sampled export used to be registered alongside it, and was the only log
here whose page visits contributed something close to a browser's real request
set. It has been removed. `data/cache` may still hold its materialized
subsample; nothing reads it.

The export carries `initiator_type` and `http_method`, so every column
`train_multi_target.engineer_features` reads means the same thing here as in
the CSV extract.
"""

from __future__ import annotations

import glob as globlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from journey import ROOT, REQUIRED_COLUMNS

EXTENSION = ROOT / "duckdb-disconnect" / "build" / "disconnect.duckdb_extension"
CACHE_DIR = ROOT / "data" / "cache"


@dataclass(frozen=True)
class Export:
    """One parquet export and how much of it the journey suite reads.

    `page_modulus` keeps 1 page domain in that many. It is set per export to
    land it near 100k requests, so a dataset costs about what the CSV extract
    costs to run and their journey errors are comparable rather than confounded
    with sample size.

    `cache_modulus`, when set, is a coarser page filter whose result is
    materialized under `data/cache`. No registered export needs it today; it is
    here for one large enough that reading whole pages means touching every
    file, where the scan takes minutes — fine once, not fine per test session.
    Filtering the cache down to `page_modulus` afterwards is exact as long as
    `page_modulus` is a multiple of `cache_modulus`, since `h % (k*m) == 0`
    implies `h % m == 0`.
    """

    glob: str
    page_modulus: int
    cache_modulus: int | None = None

    @property
    def cache(self) -> Path | None:
        if self.cache_modulus is None:
            return None
        name = Path(self.glob).parent.name
        return CACHE_DIR / f"{name}-pagemod{self.cache_modulus}.parquet"

    def __post_init__(self) -> None:
        if self.cache_modulus is not None:
            assert self.page_modulus % self.cache_modulus == 0, (
                f"page_modulus {self.page_modulus} must be a multiple of "
                f"cache_modulus {self.cache_modulus} for the cache to be exact")


EXPORTS: dict[str, Export] = {
    # 4.51M of 28.5M requests are trackers; 1 page in 45 is ~101k of them.
    "http_archive": Export(
        glob=str(ROOT / "data" / "http-archive-urls-1pct" / "*.parquet"),
        page_modulus=45),
}

# What the cache holds: the export's own columns for the pages it keeps, not
# derived features, so `_FEATURES` can change without a re-scan.
_CACHE_QUERY = """
COPY (
    SELECT url, page_domain, resource_type, transfer_bytes
    FROM read_parquet($glob)
    WHERE is_tracker(url)
      AND transfer_bytes IS NOT NULL
      AND md5_number_lower(page_domain) % $modulus = 0
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""

# Feature derivations transcribed from `sql/06_large_scale.sql`, so a column
# means the same thing in every dataset. Only the dialect changes: BigQuery's
# REGEXP_EXTRACT yields NULL on no match where DuckDB's yields '', hence the
# `nullif`s. `num_query_params` keeps the original's off-by-one — splitting an
# empty query on '&' gives one element — because the shipped model was fitted
# against that definition.
#
# `md5_number_lower` stands in for `journey._page_bucket`: a hash that is
# stable across processes and DuckDB versions, so the subsample is the same
# from run to run. `{where}` carries the tracker filter when reading an export
# directly; a cache has already had it applied.
_FEATURES = """
SELECT
    disconnect_url_host(url)                                   AS tracker_domain,
    nullif(regexp_extract(url, 'https?://[^/]+(/[^?#]*)', 1), '')
                                                               AS url_path,
    len(str_split(coalesce(nullif(regexp_extract(url, 'https?://[^/]+(/[^?#]*)', 1), ''), '/'), '/')) - 1
                                                               AS path_depth,
    lower(nullif(regexp_extract(url, '\\.([a-zA-Z0-9]+)(\\?|#|$)', 1), ''))
                                                               AS file_extension,
    regexp_matches(url, '\\?')                                 AS has_query_params,
    length(url)                                                AS url_length,
    len(str_split(regexp_extract(url, '\\?(.*)$', 1), '&'))     AS num_query_params,
    resource_type,
    initiator_type,
    http_method,
    page_domain,
    greatest(transfer_bytes, 0)                                AS transfer_bytes
FROM read_parquet($glob)
WHERE {where}
  AND md5_number_lower(page_domain) % $modulus = 0
"""


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.load_extension(str(EXTENSION))
    return con


def build_cache(export: Export, force: bool = False) -> Path:
    """Materialize `export`'s coarse page subsample, if it is not there yet.

    Returns the cache path. Delete the file (or pass `force`) to rebuild it
    after changing the tracker filter or the export itself; nothing here
    detects that for you, because a staleness check would mean the full scan
    the cache exists to avoid.
    """
    cache = export.cache
    assert cache is not None, "export has no cache_modulus"
    if cache.exists() and not force:
        return cache

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial.parquet")
    con = _connect()
    try:
        con.execute(_CACHE_QUERY.format(out=partial.as_posix()),
                    {"glob": export.glob, "modulus": export.cache_modulus})
    finally:
        con.close()
    # Renamed only on success, so an interrupted scan cannot leave a cache that
    # looks complete.
    partial.replace(cache)
    return cache


def load_request_log(export: Export | str = "http_archive") -> pd.DataFrame:
    """Load an export's page-level subsample, ready for `journey`.

    Applies the same cleaning `journey.load_request_log` does — drop a missing
    target, clip negative sizes — and returns exactly `REQUIRED_COLUMNS`, so
    the frame is interchangeable with the CSV one everywhere downstream.
    """
    if isinstance(export, str):
        export = EXPORTS[export]

    if export.cache_modulus is None:
        source, where = export.glob, "is_tracker(url)"
    else:
        source, where = str(build_cache(export)), "true"

    con = _connect()
    try:
        requests = con.execute(
            _FEATURES.format(where=where),
            {"glob": source, "modulus": export.page_modulus}).fetch_df()
    finally:
        con.close()
    return requests[list(REQUIRED_COLUMNS)]


def available(export: Export | str) -> str | None:
    """Why `export` cannot be read, or None if it can."""
    if isinstance(export, str):
        export = EXPORTS[export]
    if not EXTENSION.exists():
        return (f"{EXTENSION.relative_to(ROOT)} not built; run `make` in "
                f"duckdb-disconnect/ to run the journey tests over the parquet exports")
    cache = export.cache
    if cache is not None and cache.exists():
        return None
    if not globlib.glob(export.glob):
        return (f"no parquet files match {export.glob} (gitignored); re-run the "
                f"HTTP Archive export to enable this dataset")
    return None
