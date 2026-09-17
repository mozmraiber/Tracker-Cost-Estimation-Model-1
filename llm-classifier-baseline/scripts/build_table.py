#!/usr/bin/env python
"""Fit the size table `estimate_size` looks up and emit `src/table.rs`.

The estimator is a hierarchy of conditional *means* over the blocked-request
log, keyed on what `estimate_size` can see at call time: the Disconnect list
index of the tracker, the request context, and features of the URL itself.
Means, not medians -- the product question is "how many bytes did ETP save this
week", a sum, and summing conditional medians of a distribution that is 48%
zeros understates it badly.

Levels are tried most specific first (see LEVELS). Each level's estimate is
shrunk toward its parent's by a Laplace weight, so a path seen twice nudges the
estimate rather than replacing it, and entries compete for a fixed budget by
how many bytes of bias each one corrects.

Only URLs the Disconnect list matches are in scope, so the log is joined
against `disconnect.tracker_index` and unmatched hosts are dropped. Note that
this removes www.googletagmanager.com, which is 57% of blocked bytes in the raw
log but absent from the list; within scope the heavy hitters are
connect.facebook.net, the Instagram/Facebook media CDNs and Google's ad tags.

Usage:  python scripts/build_table.py [--budget N] [--csv PATH]
"""

from __future__ import annotations

import argparse
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
OUT = CRATE / "src" / "table.rs"

# `tests/test_browsing_journey.py` caps the built extension at 50 KB over its
# baseline. At 6 bytes an entry (u32 key + u16 value) this budget leaves room
# for the code; raise it only after re-checking that test.
DEFAULT_BUDGET = 5600

# Lookup order, most specific first. Must match `Level` in lib.rs.
L_PATH, L_TPL, L_PFX, L_EXT_Q, L_EXT, L_IDX, L_CTX = range(7)
LEVELS = [L_PATH, L_TPL, L_PFX, L_EXT_Q, L_EXT, L_IDX, L_CTX]
LEVEL_NAMES = {L_PATH: "exact path", L_TPL: "path template",
               L_PFX: "path prefix + ext", L_EXT_Q: "ext + query length",
               L_EXT: "ext", L_IDX: "tracker", L_CTX: "context"}
# The two coarsest levels are small and cover every request, so they are kept
# whole rather than made to compete for the budget.
ALWAYS = (L_CTX, L_IDX)

# Values are stored as log1p(bytes) * SCALE in a u16: a 0.024% relative step,
# and log1p(4 MB) * 4096 still fits in 16 bits.
SCALE = 4096.0
# Laplace weight pulling a group toward its parent level, as in
# `src/model/smoothed_lut.py`.
SHRINK = 4.0
# A level below `ALWAYS` needs this many training rows before it is a candidate.
MIN_COUNT = 2
# Longer "extensions" are hashes and query junk, not file types.
MAX_EXT = 8
# Query lengths are log2-bucketed and clamped here; also the cap in lib.rs.
MAX_QBUCKET = 12

RESOURCE_CONTEXT = {"script": 0, "image": 1, "video": 2}   # everything else: 3

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
    head = f"{level}|" + frame.idx.astype(str) + "|" + frame.ctx.astype(str)
    if level == L_CTX:
        return f"{level}|" + frame.ctx.astype(str)
    if level == L_IDX:
        return head
    if level == L_EXT:
        return head + "|" + frame.ext
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

def load_log(csv: Path) -> pd.DataFrame:
    """Blocked requests that the Disconnect list matches, with URL features."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv_auto('{csv}', sample_size=200000)")

    hosts = con.execute("SELECT DISTINCT tracker_domain FROM raw").df()
    hosts["idx"] = [disconnect.tracker_index(f"https://{h}/") for h in hosts.tracker_domain]
    matched = hosts[hosts.idx.notna()].astype({"idx": int})
    con.register("matched", matched)

    frame = con.execute("""
        SELECT m.idx,
               raw.url_path,
               CASE raw.resource_type WHEN 'script' THEN 0 WHEN 'image' THEN 1
                                      WHEN 'video'  THEN 2 ELSE 3 END        AS ctx,
               -- url_length counts the whole URL, so what is left once the
               -- 'https://', host and path are removed is the query.
               greatest(raw.url_length - length(raw.tracker_domain)
                        - length(raw.url_path) - 8, 0)                        AS query_length,
               greatest(raw.transfer_bytes, 0)                                AS y
        FROM raw JOIN matched m USING (tracker_domain)
        WHERE raw.transfer_bytes IS NOT NULL
    """).df()

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
    frame["qbucket"] = query_bucket(frame.query_length.to_numpy())
    return frame


def fit(log: pd.DataFrame, budget: int) -> tuple[dict[str, int], float, pd.DataFrame]:
    """Shrink each level toward its parent, then spend `budget` on the best."""
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

    competing = pd.concat([stats[lv].assign(level=lv) for lv in LEVELS if lv not in ALWAYS])
    chosen = competing.sort_values("gain", ascending=False).head(max(0, budget - len(table)))
    table.update(zip(chosen.index, quantize(chosen.est.to_numpy()).tolist()))
    return table, float(y.mean()), chosen


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

def render(entries: dict[int, int], global_mean: float, source: str) -> str:
    keys = sorted(entries)
    lines = [
        "// @generated by scripts/build_table.py -- do not edit by hand.",
        f"// Source: {source}",
        "//",
        "// Conditional mean transfer size per lookup key, as",
        "// `log1p(bytes) * VALUE_SCALE` rounded into a u16. Keys are the FNV-1a",
        "// hashes `key_of` builds; KEYS is sorted so lookups can binary search,",
        "// and VALUES is parallel to it.",
        "",
        "/// Fixed-point scale of the stored values.",
        "pub const VALUE_SCALE: f64 = %r;" % SCALE,
        "",
        "/// Mean size over every in-scope request; the answer when even the",
        "/// context level has no entry.",
        "pub const GLOBAL_MEAN: u16 = %d;" % int(quantize(np.array([global_mean]))[0]),
        "",
        "pub static KEYS: [u32; %d] = [" % len(keys),
    ]
    lines += _columns(f"0x{k:08x}," for k in keys)
    lines += ["];", "", "pub static VALUES: [u16; %d] = [" % len(keys)]
    lines += _columns(f"{entries[k]}," for k in keys)
    lines += ["];", ""]
    return "\n".join(lines)


def _columns(items, per_line: int = 8) -> list[str]:
    items = list(items)
    return ["    " + " ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="maximum table entries (6 bytes each)")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"{args.csv} not present (gitignored); regenerate it with the extract in sql/")

    log = load_log(args.csv)
    print(f"{len(log):,} in-scope requests over {log.idx.nunique()} Disconnect entries, "
          f"mean {log.y.mean():,.0f} bytes")

    table, global_mean, chosen = fit(log, args.budget)
    entries = resolve_collisions(table)
    counts = chosen.level.value_counts()
    print("entries by level: " + ", ".join(
        f"{LEVEL_NAMES[lv]}={int(counts.get(lv, 0))}" for lv in LEVELS if lv not in ALWAYS))
    print(f"plus {len(table) - len(chosen)} always-kept tracker/context entries")
    print(f"{len(entries)} entries -> {len(entries) * 6:,} bytes of table")

    OUT.write_text(render(entries, global_mean, f"{args.csv.name}, {disconnect.list_info()['source']}"))
    print(f"wrote {OUT.relative_to(ROOT)}")
    subprocess.run(["rustfmt", str(OUT)], check=False)


if __name__ == "__main__":
    main()
