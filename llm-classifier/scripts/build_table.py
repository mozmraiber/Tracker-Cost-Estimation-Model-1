#!/usr/bin/env python
"""Fit the size table `estimate_resources` looks up and emit `src/table.rs`.

The estimator is a hierarchy of conditional *means* over the blocked-request
log, keyed on what `estimate_resources` can see at call time: the request context and
features of the URL itself. Means, not medians -- the product question is "how
many bytes did ETP save this week", a sum, and summing conditional medians of a
distribution that is 48% zeros understates it badly.

Which tracker served the request is deliberately not a key. The log is still
restricted to Disconnect-matched hosts, because those are the requests ETP
blocks, but the asset path identifies the vendor well enough on its own that
splitting entries per tracker costs more budget than it returns -- see the
measurements against LEVELS below.

`estimate_resources` also receives the request's initiator and its HTTP method.
Only one of the two is fitted here, and the measurements are in BODYLESS_METHODS
and INITIATOR_UNUSED: a method whose response carries no body is worth answering
from the method alone, and the initiator is not worth a key at this budget.

Levels are tried most specific first (see LEVELS), and entries compete for a
fixed budget by how many bytes of bias each one corrects against the next level
up. The budget is a size budget, not a statistical one -- accuracy is still
climbing where it runs out -- so the table is packed to 4 bytes an entry to buy
as many as the extension's 50 KB allows. See `pack`.

Only URLs `disconnect.is_tracker` calls trackers are in scope, and unmatched
hosts are dropped. That excludes the Disconnect `Content` category -- a
tracking company's own first-party properties, which a browser does not block
-- and it excludes www.googletagmanager.com, 57% of blocked bytes in the raw
log but absent from the list entirely. Within scope the heavy hitters are
connect.facebook.net, the Instagram/Facebook media CDNs and Google's ad tags.

Either log can be fitted. `--csv` reads the BigQuery extract, which records
URL features rather than URLs, so the query length is reconstructed from
`url_length`. `--parquet` reads the HTTP Archive export, which carries whole
URLs: it is 50x larger, is filtered here rather than at export time, and its
query lengths are exact. Both feed the same fit.

Usage:  python scripts/build_table.py [--budget N] [--csv PATH | --parquet GLOB]
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import disconnect
except ImportError:  # pragma: no cover - the generator needs the extension
    sys.exit("need the `disconnect` extension importable; build duckdb-disconnect first")

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "llm-classifier"
DEFAULT_CSV = ROOT / "data" / "raw" / "per_request_1pct.csv"
DEFAULT_PARQUET = str(ROOT / "data" / "http-archive-urls-1pct" / "*.parquet")
# The parquet log is filtered with the DuckDB build of the matcher rather than
# the Python one, so `is_tracker` runs inside the scan over 28M rows.
DUCKDB_EXTENSION = ROOT / "duckdb-disconnect" / "build" / "disconnect.duckdb_extension"
OUT = CRATE / "src" / "table.rs"

# `tests/test_browsing_journey.py` caps the extension `llm-classifier-python`
# builds over this crate at 50 KB over its baseline, but the real wall is coarser than that number suggests: the dylib is
# laid out in 16 KB pages, so the measured size does not creep up with the table,
# it sits flat and then jumps a whole page. On this toolchain the jump lands
# between 8,800 and 8,900 entries and clears the cap by 2 KB, so ~8,800 is the
# most that fits and 34.7 KB is the size it actually reports.
#
# That ceiling fell from ~9,600 when `RequestContext` went from 4 variants to 11:
# each one costs pyo3 registration code, and seven of them came to about 800
# entries' worth of it. Adding `RequestInitiator` cost 48 bytes and did not
# move it at all -- the four variants' registration code fit in the slack of
# the page the table was already in, and the measured size is still flat from
# 8,700 entries to 8,800. Splitting the extension out into its own crate cost
# nothing: the `.so` came out byte for byte the size it was. Hence also `pack`: at the 6 bytes an entry a flat
# u32/u16 table costs, the same page would hold only ~5,200. Before raising this,
# re-measure -- build, then compare the two .so sizes the way that test does.
DEFAULT_BUDGET = 8700

# Keys are bucketed by their top byte, so an entry stores only the low 24 bits.
N_BUCKETS = 256
# The value byte indexes this many codebook entries.
CODEBOOK_SIZE = 256

# Lookup order, most specific first. Must match `Level` in lib.rs.
#
# There is no per-tracker rung, and no level keys on the tracker: `estimate_resources`
# is not told which Disconnect entry matched. Adding it back is worth 0.7 points
# of median journey error (7.8% -> 7.1%), which does not pay for making the
# caller resolve the tracker first.
#
# The request *host* is keyed on, at L_HOST and L_HOST_TPL. A flat host rung
# was worth -1.3 points (9.1%) when it competed with the path rungs on 5,213
# exact hosts; placed below them, and templated so per-customer subdomains
# pool, it is flat on the journey suite and worth 3.6 points of aggregate bias
# on the out-of-sample top-500 crawl (-6.2% -> -2.6%). The journey suite scores
# the log this table is fitted on, where paths recur and the host adds nothing;
# `tests/test_top500_estimates.py` is what sees the difference.
L_PATH, L_TPL, L_PFX, L_HOST, L_HOST_TPL, L_EXT_Q, L_EXT, L_CTX = range(8)
LEVELS = [L_PATH, L_TPL, L_PFX, L_HOST, L_HOST_TPL, L_EXT_Q, L_EXT, L_CTX]
LEVEL_NAMES = {L_PATH: "exact path", L_TPL: "path template",
               L_PFX: "path prefix + ext", L_HOST: "request host",
               L_HOST_TPL: "host template", L_EXT_Q: "ext + query length",
               L_EXT: "ext", L_CTX: "context"}
# `L_CTX` is fitted like any other level -- it is the root of the parent chain,
# so the levels above it are ranked against what a miss would actually return --
# but it is not stored as table entries. It ships as CONTEXT_FALLBACK, one mean
# per context, which `estimate` indexes directly. That covers every context
# without spending budget or risking a lookup miss, so nothing is always-kept.
TABLE_LEVELS = [L_PATH, L_TPL, L_PFX, L_HOST, L_HOST_TPL, L_EXT_Q, L_EXT]
ALWAYS: tuple[int, ...] = ()

# Values are stored as log1p(bytes) * SCALE in a u16: a 0.024% relative step,
# and log1p(4 MB) * 4096 still fits in 16 bits.
SCALE = 4096.0
# Laplace weight pulling a group toward its parent level, as in
# `src/model/smoothed_lut.py`. Zero here, which is not what that module wants:
# a given asset's transfer size is near-deterministic -- the same script is the
# same bytes every fetch -- so a group mean is already precise at n=1 and
# shrinking it only adds bias. Measured on the journey suite, every step up from
# 0 costs accuracy (0 -> 5.6%, 0.1 -> 6.3%, 1.0 -> 9.9% median error). The
# parent chain still does the work that matters, in `gain` below. Raise this if
# the table is ever fitted on a log small enough for group means to be noisy.
SHRINK = 0.0
# A level below `ALWAYS` needs this many training rows before it is a candidate.
# One: a path seen once still pins that asset's size exactly, and an entry that
# does not pay for itself is dropped by the `gain` ranking anyway rather than by
# a count threshold. Admitting singletons is what lets the specific levels cover
# the long tail -- it moves p90 journey error from 56% to 26%.
MIN_COUNT = 1
# Methods whose response carries no payload, so the URL has nothing to say
# about its size. `estimate` answers these from BODYLESS_MEAN below and never
# reaches the table, and they are held out of the level fit -- a CORS preflight
# shares the path of the POST it precedes, so leaving them in drags that path's
# mean toward zero for the requests that do transfer something.
#
# HEAD is bodyless by specification. OPTIONS is not, but in this log it always
# is: HEAD averages exactly 0.00 bytes over 2,514 requests and OPTIONS 9.46
# over 63,302, every one of them a preflight. Between them that is 1.37% of the
# log, and the table answered them with a mean of 1,262 bytes -- the path each
# preflight shares with the POST it precedes. Per-request MAE over those rows
# falls 84x, from 1,256 bytes to 15; on the two test halves the journey suite
# scores, 163x and 89x.
#
# Those requests are 0.0017% of all blocked bytes, so this is a fix to
# individual answers rather than to a weekly total, and the journey suite says
# so. Over seeds 3, 5 and 7 it is worth 0.05 points of median journey error,
# 0.3 points of the CSV extract's p90 (15.25% -> 14.95%), +0.003 on the
# within-25% rate, and 1.5% and 0.7% of per-request MAE on the two datasets.
# The rule itself accounts for the p90 and about two thirds of the MAE;
# holding these rows out of the level fit accounts for the rest, and for the
# CSV median.
#
# POST is deliberately not here. Its responses are mostly empty too -- 85 bytes
# on average -- but the URL predicts them better than the method does (MAE 65
# against 132 for a per-(method, context) mean), because a beacon endpoint has
# a path of its own. So POST keeps using the table.
BODYLESS_METHODS = ("HEAD", "OPTIONS")

# What keying on `initiator_type` was measured to be worth: nothing, at this
# budget. Fitted as extra rungs above the URL levels -- so a request with a
# known initiator tries an initiator-keyed key first and falls through to
# today's -- it moved the median journey error by less than 0.2 points either
# way, which is inside the 1-2 points a change of journey seed moves it, while
# costing 1-2 points of the within-25% rate on the HTTP Archive export, whose
# entries it displaces (0.948 -> 0.937 keyed at the two coarsest levels, 0.928
# keyed from the prefix down). Keying on the initiator and the method together
# was no better. Those came from a prototype fit scored in Python rather than
# through the built extension, which would need matching levels in lib.rs; it
# tracked the shipped pipeline to a few tenths of a point.
#
# The reason is that the path already carries it. Script-initiated GETs average
# 17.2 KB and parser-discovered ones 3.5 KB, a 5x split, and the table predicts
# them at 17.3 KB and 3.7 KB *without being told which is which* -- the big
# script-initiated assets are bundles and media with paths of their own. The
# same held for the tracker identity (see LEVELS) and for the same reason.
#
# So `estimate_resources` takes an initiator, for parity with
# `xgb-classifier`'s interface, and this generator does not key on it. That
# crate does use it, and there it is worth 19 points: a booster reading 80
# features has no path-level table to carry the information instead.
INITIATOR_UNUSED = True

# Longer "extensions" are hashes and query junk, not file types.
MAX_EXT = 8
# Query lengths are log2-bucketed and clamped here; also the cap in lib.rs.
MAX_QBUCKET = 12

# `resource_type` in the log -> `RequestContext` discriminant in lib.rs. Keep
# the two in step; the discriminant is hashed into every key.
#
# The log's `json` has no variant of its own and lands in OTHER, as does anything
# else the crawl records that is not named here. `wasm` is the other way round: a
# variant with no rows in this log, which is why its fallback comes out as the
# global mean.
CONTEXT_OTHER = 3
RESOURCE_CONTEXT = {
    "script": 0, "image": 1, "video": 2, "other": CONTEXT_OTHER, "audio": 4,
    "css": 5, "font": 6, "html": 7, "text": 8, "wasm": 9, "xml": 10,
}
# Length of CONTEXT_FALLBACK: one slot per `RequestContext` variant.
N_CONTEXTS = max(RESOURCE_CONTEXT.values()) + 1

# Contexts are folded into these groups before being keyed on. Only the URL
# levels fold -- CONTEXT_FALLBACK stays per-context, since an array of 11 costs
# nothing and a font is not a video.
#
# Keying all 11 separately measures the same as folding to these 4 (7.72% median
# journey error either way, over 8 seeds), because the extension and path already
# carry most of what the context would say. What folding buys is the contexts the
# log barely covers: `xml` has 2,342 rows and wins *no* table entry of its own
# under the gain ranking, so every xml request would be answered by a context
# mean whatever its URL -- folding puts it on `doc`'s entries and takes its
# coverage from 0% to 90%. `audio` and `wasm`, which the log has no rows for at
# all, likewise get to ride on their group instead of always falling back.
CONTEXT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code",  ("script", "css", "wasm")),
    ("media", ("image", "video", "audio", "font")),
    ("doc",   ("html", "text", "xml")),
    ("other", ("other",)),
)


def _key_group_table() -> list[int]:
    """Context discriminant -> key group, as lib.rs reads it from `KEY_GROUP`."""
    groups = [0] * N_CONTEXTS
    for group, names in enumerate(CONTEXT_GROUPS):
        for name in names[1]:
            groups[RESOURCE_CONTEXT[name]] = group
    return groups


KEY_GROUP = _key_group_table()

_HEXISH = re.compile(r"^[0-9a-f-]{8,}$", re.IGNORECASE)
_TOKEN = re.compile(r"[0-9a-z]+", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# URL features. Every rule here has a twin in lib.rs; keep them in step.
# --------------------------------------------------------------------------- #

def extension(path: str) -> str:
    """Text after the last dot of the last path segment, lowercased.

    Empty when the last segment has no dot or the tail is too long to be a real
    file type, which is what the log's own `file_extension` column amounts to.
    """
    segment = path.rsplit("/", 1)[-1]
    dot = segment.rfind(".")
    if dot < 0:
        return ""
    ext = segment[dot + 1:].lower()
    return ext if 0 < len(ext) <= MAX_EXT else ""


def _normalize_segment(segment: str) -> str:
    """Collapse the version and hash parts of one path segment to '#'.

    Tracker assets are cache-busted: `/pagead/managed/js/adsense/m202608060101/
    show_ads_impl_fy2021.js` and `/modules.75bf3488a101739acfe9.js` are the same
    asset every week under a new name. Templating them keeps one table entry
    that still matches after the vendor rolls a new build, instead of spending
    one entry per build and missing next week's.
    """
    if _HEXISH.match(segment) and any(c.isdigit() for c in segment):
        return "#"

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isdigit():
            return "#" if len(token) >= 4 else token
        return "#" if len(token) >= 8 and any(c.isdigit() for c in token) else token

    return _TOKEN.sub(replace, segment)


def normalize_path(path: str) -> str:
    return "/".join(_normalize_segment(s) for s in path.split("/"))


def normalize_host(host: str) -> str:
    """Collapse the generated labels of a hostname, as `normalize_path` does.

    Ad and data vendors give each customer their own subdomain under a stable
    suffix: `d39f98ec-9259-4f8b-896d-7ab58be1f900.edge.permutive.app` and
    twenty more like it are one asset served per publisher, averaging 179 KB
    where the per-context script mean is 31 KB. Keyed on the exact host, every
    UUID the training log never saw misses; keyed on `#.edge.permutive.app`
    they pool, and an unseen customer still resolves.

    Deliberately the same `_normalize_segment` the path rungs use rather than a
    hostname rule of its own, so there is one definition of "this label was
    generated" to keep in step with lib.rs. It leaves real labels alone,
    including dashed ones like `go-mpulse` -- `_HEXISH` needs a digit.
    """
    return ".".join(_normalize_segment(label) for label in host.split("."))


def path_prefix(template: str) -> str:
    """The first two segments of the templated path: the asset family."""
    return "/".join(template.split("/")[:3])


def query_bucket(query_length: np.ndarray) -> np.ndarray:
    """log2 bucket of the query-string length, including the '?'.

    A coarse stand-in for the transform parameters the media CDNs carry
    (`stp=dst-jpg_e35_s640x640`), which the log does not record verbatim.
    """
    lengths = np.nan_to_num(np.asarray(query_length, dtype=float), nan=0.0)
    return np.minimum(np.log2(np.maximum(lengths, 0) + 1).astype(int), MAX_QBUCKET)


# --------------------------------------------------------------------------- #
# Keys. The strings built here are exactly the bytes lib.rs hashes.
# --------------------------------------------------------------------------- #

def key_strings(frame: pd.DataFrame, level: int) -> pd.Series:
    """Key for every row at one level.

    The URL levels key on the folded `kgroup`, not the context itself; only the
    context level, which ships as CONTEXT_FALLBACK rather than as table entries,
    keys on the context. See CONTEXT_GROUPS.
    """
    head = f"{level}|" + frame.kgroup.astype(str)
    if level == L_CTX:
        return f"{level}|" + frame.ctx.astype(str)
    if level == L_EXT:
        return head + "|" + frame.ext
    if level == L_HOST:
        return head + "|" + frame.host
    if level == L_HOST_TPL:
        return head + "|" + frame.host_template
    if level == L_EXT_Q:
        return head + "|" + frame.ext + "|" + frame.qbucket.astype(str)
    if level == L_PFX:
        return head + "|" + frame.ext + "|" + frame.prefix
    if level == L_TPL:
        return head + "|" + frame.template
    if level == L_PATH:
        return head + "|" + frame.url_path
    raise ValueError(level)


def fnv1a32(data: bytes) -> int:
    """FNV-1a 64 folded to 32 bits, as `Hasher::finish` in lib.rs does."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return ((h >> 32) ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF


def quantize(values: np.ndarray) -> np.ndarray:
    scaled = np.round(np.log1p(np.maximum(np.asarray(values, dtype=float), 0.0)) * SCALE)
    return np.minimum(scaled, 65535).astype(np.uint16)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def context_case_sql(column: str = "raw.resource_type") -> str:
    """`RESOURCE_CONTEXT` as a SQL CASE, so the mapping is written down once."""
    whens = " ".join(f"WHEN '{name}' THEN {code}"
                     for name, code in sorted(RESOURCE_CONTEXT.items()))
    return f"CASE {column} {whens} ELSE {CONTEXT_OTHER} END"


def load_csv_log(csv: Path) -> pd.DataFrame:
    """In-scope requests from the BigQuery extract, with URL features.

    The extract has no URL column, so the query length is what is left of
    `url_length` once the 'https://', host and path are subtracted. It is an
    approximation -- an http:// request is over-counted by one -- but it is the
    only query signal this log carries.
    """
    ctx_case = context_case_sql()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv_auto('{csv}', sample_size=200000)")

    hosts = con.execute("SELECT DISTINCT tracker_domain FROM raw").df()
    urls = [f"https://{h}/" for h in hosts.tracker_domain]
    hosts["entry"] = [disconnect.tracker_pattern(u) if disconnect.is_tracker(u) else None
                      for u in urls]
    con.register("matched", hosts[hosts.entry.notna()])

    frame = con.execute(f"""
        SELECT m.entry,
               lower(raw.tracker_domain)                                      AS host,
               raw.url_path,
               {ctx_case}                                                     AS ctx,
               greatest(raw.url_length - length(raw.tracker_domain)
                        - length(raw.url_path) - 8, 0)                        AS query_length,
               raw.http_method,
               greatest(raw.transfer_bytes, 0)                                AS y
        FROM raw JOIN matched m USING (tracker_domain)
        WHERE raw.transfer_bytes IS NOT NULL
    """).df()
    return derive_features(frame)


def load_parquet_log(glob: str) -> pd.DataFrame:
    """In-scope requests from the HTTP Archive export, with URL features.

    Filtered by the DuckDB build of the matcher inside the scan, because the
    export is unfiltered: 28M requests in, 4.5M trackers out. Whole URLs are
    recorded here, so the path and the query length are read off directly
    rather than reconstructed -- the same values `estimate_resources` will compute
    from the URL at call time.
    """
    ctx_case = context_case_sql("resource_type")
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    try:
        con.load_extension(str(DUCKDB_EXTENSION))
        frame = con.execute(f"""
            SELECT tracker_pattern(url)                                       AS entry,
                   lower(regexp_extract(url, 'https?://([^/?#]+)', 1))        AS host,
                   coalesce(nullif(regexp_extract(url, 'https?://[^/]+(/[^?#]*)', 1), ''), '/')
                                                                              AS url_path,
                   {ctx_case}                                                 AS ctx,
                   -- Includes the '?', which is what query_bucket() counts.
                   length(regexp_extract(url, '(\\?[^#]*)', 1))                AS query_length,
                   http_method,
                   greatest(transfer_bytes, 0)                                AS y
            FROM read_parquet($glob)
            WHERE is_tracker(url) AND transfer_bytes IS NOT NULL
        """, {"glob": glob}).df()
    finally:
        con.close()
    return derive_features(frame)


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The level keys both logs are fitted on, from `url_path` and `ctx`."""
    frame["url_path"] = frame.url_path.fillna("/")
    # Templating is the slow part and paths repeat heavily, so map it over the
    # distinct paths rather than over every row.
    paths = pd.Index(frame.url_path.unique())
    per_path = pd.DataFrame({
        "ext": [extension(p) for p in paths],
        "template": [normalize_path(p) for p in paths],
    }, index=paths)
    per_path["prefix"] = [path_prefix(t) for t in per_path.template]
    frame = frame.join(per_path, on="url_path")
    # A log with no host column (an older extract) keys every row alike, which
    # the gain ranking then declines to spend entries on.
    frame["host"] = (frame.host.fillna("").astype(str).str.lower()
                     if "host" in frame else "")
    hosts = pd.Index(frame.host.unique())
    frame = frame.join(pd.DataFrame(
        {"host_template": [normalize_host(h) for h in hosts]}, index=hosts), on="host")
    frame["qbucket"] = query_bucket(frame.query_length.to_numpy())
    frame["kgroup"] = [KEY_GROUP[c] for c in frame.ctx]
    return frame


def split_bodyless(log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate the requests a bodyless method answers from the rest.

    The first frame is what the levels are fitted on; the second is what
    BODYLESS_MEAN is fitted on. A log with no `http_method` column would put
    everything in the first, which is what this estimator did before the
    method was part of its interface.
    """
    method = log.http_method.astype("string").str.upper()
    bodyless = method.isin(BODYLESS_METHODS).fillna(False)
    return log[~bodyless], log[bodyless]


def bodyless_mean(bodyless: pd.DataFrame, fallback: float) -> np.ndarray:
    """The one value `estimate` answers a bodyless method with.

    One mean over both methods rather than one each: OPTIONS outnumbers HEAD
    25 to 1 here, so the pooled figure is the preflight figure to a tenth of a
    byte, and HEAD's own mean is zero -- nine bytes is not a difference worth a
    second entry. `fallback` covers a log that recorded no bodyless request at
    all, and is only reachable through `--csv` on an extract without the
    column.
    """
    if bodyless.empty:
        print("warning: no HEAD or OPTIONS requests in the log; "
              f"BODYLESS_MEAN falls back to {fallback:,.0f} bytes")
        return quantize(np.array([fallback]))
    return quantize(np.array([bodyless.y.mean()]))


def fit(log: pd.DataFrame,
        budget: int) -> tuple[dict[str, int], np.ndarray, pd.DataFrame]:
    """Shrink each level toward its parent, then spend `budget` on the best.

    Returns the table, the per-context fallback means, and the chosen entries.
    """
    y = log.y.to_numpy()
    parent = np.full(len(log), y.mean())
    stats: dict[int, pd.DataFrame] = {}

    for level in reversed(LEVELS):          # coarse first: parents before children
        keys = key_strings(log, level)
        grouped = (pd.DataFrame({"k": keys.to_numpy(), "y": y, "parent": parent})
                   .groupby("k")
                   .agg(n=("y", "size"), mean=("y", "mean"), parent=("parent", "mean")))
        grouped = grouped[grouped.n >= (1 if level in ALWAYS else MIN_COUNT)]
        grouped["est"] = ((grouped.n * grouped["mean"] + SHRINK * grouped.parent)
                          / (grouped.n + SHRINK))
        # What a journey total cares about: the bytes of bias this entry fixes.
        grouped["gain"] = grouped.n * (grouped.est - grouped.parent).abs()
        stats[level] = grouped
        parent = keys.map(grouped.est).fillna(pd.Series(parent, index=keys.index)).to_numpy()

    table: dict[str, int] = {}
    for level in ALWAYS:
        table.update(zip(stats[level].index, quantize(stats[level].est.to_numpy()).tolist()))

    competing = pd.concat([stats[lv].assign(level=lv) for lv in TABLE_LEVELS])
    chosen = competing.sort_values("gain", ascending=False).head(max(0, budget - len(table)))
    table.update(zip(chosen.index, quantize(chosen.est.to_numpy()).tolist()))
    return table, context_fallback(stats[L_CTX], float(y.mean())), chosen


def context_fallback(context_stats: pd.DataFrame, global_mean: float) -> np.ndarray:
    """The context level as an array, one mean per `RequestContext` variant.

    This is what `estimate` answers with when no URL level matched. A context
    the log never recorded -- `wasm` here -- has nothing of its own to average,
    so it takes the mean over every in-scope request instead.
    """
    means = np.full(N_CONTEXTS, global_mean)
    for key, est in context_stats.est.items():
        means[int(key.split("|")[1])] = est
    return quantize(means)


def resolve_collisions(table: dict[str, int]) -> dict[int, int]:
    """Hash to u32 and drop the rare pair that lands on the same slot.

    A collision between two stored keys would answer one of them with the
    other's size, so the loser is dropped and its lookup falls through to the
    next level instead.
    """
    hashed: dict[int, tuple[str, int]] = {}
    dropped = 0
    for key, value in table.items():
        h = fnv1a32(key.encode())
        if h in hashed and hashed[h][0] != key:
            dropped += 1
            continue
        hashed[h] = (key, value)
    if dropped:
        print(f"dropped {dropped} colliding entr{'y' if dropped == 1 else 'ies'}")
    return {h: value for h, (_, value) in hashed.items()}


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #

def build_codebook(values: np.ndarray, size: int = CODEBOOK_SIZE) -> np.ndarray:
    """Lloyd-max codebook over the quantized values.

    An entry's value is a byte indexing this, rather than the u16 itself, which
    is what makes an entry 4 bytes instead of 5. It costs almost nothing:
    the fitted means cluster hard -- a tracker serves a handful of distinct
    asset sizes, not a continuum -- so 256 centres reproduce them to well inside
    the noise of the estimate. Measured on the journey suite the difference
    against exact u16 values at the same entry count is 0.01 points of median
    error, while the byte saved buys ~2,400 more entries, worth 1.2 points.
    """
    v = np.asarray(values, dtype=float)
    centers = np.unique(np.round(np.quantile(v, np.linspace(0, 1, size))))
    for _ in range(40):
        index = np.abs(v[:, None] - centers[None, :]).argmin(axis=1)
        moved = np.round([v[index == j].mean() if (index == j).any() else centers[j]
                          for j in range(len(centers))])
        if np.array_equal(moved, centers):
            break
        centers = moved
    return centers


def pack(entries: dict[int, int]) -> dict[str, np.ndarray]:
    """Lay the table out as `table_lookup` in lib.rs reads it.

    Keys are sorted, then split into `N_BUCKETS` buckets by their top byte, so
    only the low 24 bits need storing -- as a `KEY_HI` byte and a `KEY_LO`
    halfword, parallel arrays rather than a packed 3-byte record so neither
    needs an unaligned load. `BUCKETS` holds each bucket's start, plus a final
    total so a bucket's end is just the next entry.

    Nothing is lost: the bucket index *is* the top byte, so a lookup still
    compares the whole 32-bit hash and a miss still falls through to the next
    level.
    """
    keys = np.array(sorted(entries), dtype=np.uint32)
    if len(keys) > 65535:
        raise ValueError(f"{len(keys)} entries overflows the u16 BUCKETS offsets")
    values = np.array([entries[int(k)] for k in keys], dtype=np.float64)

    codebook = build_codebook(values)
    if len(codebook) > 256:
        raise ValueError(f"{len(codebook)} codebook entries overflows the u8 VALUES index")
    index = np.abs(values[:, None] - codebook[None, :]).argmin(axis=1)

    counts = np.bincount(keys >> 24, minlength=N_BUCKETS)
    starts = np.concatenate([[0], np.cumsum(counts)])

    return {
        "buckets": starts.astype(np.uint16),
        "key_hi": ((keys >> 16) & 0xFF).astype(np.uint8),
        "key_lo": (keys & 0xFFFF).astype(np.uint16),
        "values": index.astype(np.uint8),
        "codebook": codebook.astype(np.uint16),
    }


def render(entries: dict[int, int], fallback: np.ndarray, bodyless: np.ndarray,
           source: str) -> str:
    parts = pack(entries)
    n = len(parts["key_hi"])
    lines = [
        "// @generated by scripts/build_table.py -- do not edit by hand.",
        f"// Source: {source}",
        "//",
        "// Conditional mean transfer size per lookup key, as",
        "// `log1p(bytes) * VALUE_SCALE` rounded into a u16 and then replaced by",
        "// the nearest CODEBOOK entry, which VALUES indexes with a byte.",
        "//",
        "// Keys are the FNV-1a hashes `key_of` builds, sorted and bucketed by",
        "// their top byte: BUCKETS[b]..BUCKETS[b + 1] is the run of entries whose",
        "// key starts with byte b, and KEY_HI/KEY_LO carry the low 24 bits of",
        "// each. That is 4 bytes an entry, which is what keeps the table inside",
        "// the extension's size budget. See `table_lookup` in lib.rs.",
        "",
        "/// Fixed-point scale of the stored values.",
        "pub const VALUE_SCALE: f64 = %r;" % SCALE,
        "",
        "/// Mean size per request context, as the answer when no URL level",
        "/// matched. Indexed by the `RequestContext` discriminant.",
        "pub static CONTEXT_FALLBACK: [u16; %d] = [%s];"
        % (len(fallback), ", ".join(str(int(v)) for v in fallback)),
        "",
        "/// Mean size of a request whose method forbids a response body, which",
        "/// `estimate` answers with instead of consulting the table at all.",
        "/// Fitted over the %s requests alone; see BODYLESS_METHODS in"
        % "/".join(BODYLESS_METHODS),
        "/// scripts/build_table.py.",
        "pub const BODYLESS_MEAN: u16 = %d;" % int(bodyless[0]),
        "",
        "/// Key group each request context folds into before being hashed into a",
        "/// lookup key: %s." % ", ".join(
            f"{g} = {name}" for g, (name, _) in enumerate(CONTEXT_GROUPS)),
        "///",
        "/// Indexed by the `RequestContext` discriminant. See CONTEXT_GROUPS in",
        "/// scripts/build_table.py for why the contexts fold at all.",
        "pub static KEY_GROUP: [u8; %d] = [%s];"
        % (len(KEY_GROUP), ", ".join(str(g) for g in KEY_GROUP)),
        "",
        "/// Start of each top-byte bucket, plus the entry count as a final bound.",
        "pub static BUCKETS: [u16; %d] = [" % len(parts["buckets"]),
    ]
    lines += _columns(f"{v}," for v in parts["buckets"])
    lines += ["];", "", "/// Bits 16..24 of each key.", "pub static KEY_HI: [u8; %d] = [" % n]
    lines += _columns((f"{v}," for v in parts["key_hi"]), per_line=16)
    lines += ["];", "", "/// Bits 0..16 of each key.", "pub static KEY_LO: [u16; %d] = [" % n]
    lines += _columns(f"{v}," for v in parts["key_lo"])
    lines += ["];", "", "/// Codebook index of each entry's value.",
              "pub static VALUES: [u8; %d] = [" % n]
    lines += _columns((f"{v}," for v in parts["values"]), per_line=16)
    lines += ["];", "", "/// The distinct quantized values VALUES indexes.",
              "pub static CODEBOOK: [u16; %d] = [" % len(parts["codebook"])]
    lines += _columns(f"{v}," for v in parts["codebook"])
    lines += ["];", ""]
    return "\n".join(lines)


def _columns(items, per_line: int = 8) -> list[str]:
    items = list(items)
    return ["    " + " ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help=f"BigQuery extract (default {DEFAULT_CSV.name})")
    source.add_argument("--parquet", type=str, nargs="?", const=DEFAULT_PARQUET,
                        help="HTTP Archive export glob, filtered here by is_tracker()")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="maximum table entries (4 bytes each)")
    args = parser.parse_args()

    if args.parquet:
        if not DUCKDB_EXTENSION.exists():
            sys.exit(f"{DUCKDB_EXTENSION.relative_to(ROOT)} not built; run `make` in duckdb-disconnect")
        if not glob.glob(args.parquet):
            sys.exit(f"no parquet files match {args.parquet}")
        source_name = Path(args.parquet).parent.name
        log = load_parquet_log(args.parquet)
    else:
        csv = args.csv or DEFAULT_CSV
        if not csv.exists():
            sys.exit(f"{csv} not present (gitignored); regenerate it with the extract in sql/")
        source_name = csv.name
        log = load_csv_log(csv)

    print(f"{len(log):,} in-scope requests over {log.entry.nunique()} Disconnect entries, "
          f"mean {log.y.mean():,.0f} bytes")

    log, bodyless = split_bodyless(log)
    if not bodyless.empty:
        by_method = bodyless.groupby(bodyless.http_method.str.upper()).y.agg(["size", "mean"])
        print(f"{len(bodyless):,} requests held out for BODYLESS_MEAN "
              f"({bodyless.y.mean():,.2f} bytes): " + ", ".join(
                  f"{m}={int(r['size']):,}@{r['mean']:,.2f}B"
                  for m, r in by_method.iterrows()))
    bodyless_value = bodyless_mean(bodyless, float(log.y.mean()))

    table, fallback, chosen = fit(log, args.budget)
    entries = resolve_collisions(table)
    counts = chosen.level.value_counts()
    print("entries by level: " + ", ".join(
        f"{LEVEL_NAMES[lv]}={int(counts.get(lv, 0))}" for lv in TABLE_LEVELS))
    print(f"{len(entries)} entries -> {len(entries) * 4 + 2 * (N_BUCKETS + 1)
                                          + 2 * CODEBOOK_SIZE:,} bytes of table")
    by_code = {code: name for name, code in RESOURCE_CONTEXT.items()}
    print("context fallback: " + ", ".join(
        f"{by_code.get(c, c)}={np.expm1(v / SCALE):,.0f}B" for c, v in enumerate(fallback)))
    print("key groups:       " + ", ".join(
        f"{name}={'/'.join(names)}" for name, names in CONTEXT_GROUPS))

    OUT.write_text(render(entries, fallback, bodyless_value,
                          f"{source_name}, {disconnect.list_info()['source']}"))
    print(f"wrote {OUT.relative_to(ROOT)}")
    subprocess.run(["rustfmt", str(OUT)], check=False)


if __name__ == "__main__":
    main()
