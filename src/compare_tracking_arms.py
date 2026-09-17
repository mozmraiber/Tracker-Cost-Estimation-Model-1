"""
Join the two arms of the paired tracking crawl into per-page savings, and
recover ground-truth byte sizes for the requests Firefox blocked.

Reads the two output directories produced by firefox_crawl_500_tracking.py
(--mode normal and --mode private) and writes three tables:

  paired_pages.csv
      One row per page present in both arms: request counts, transfer bytes
      and CPU seconds on each side, and the deltas. This is the "what did
      blocking save on this page" table.

  blocked_observed_bytes.csv
      One row per request the blocking arm refused, annotated with the byte
      size that same request actually transferred in the control arm. This is
      the label the cost model wants: a blocked tracker request paired with
      what it would really have cost. Rows that could not be matched are kept
      with match_kind=unmatched rather than dropped, so the match rate is
      visible instead of silently inflating the mean.

  compare_summary.json
      Run-level totals, match rates, and the caveats worth carrying into the
      paper.

MATCHING, AND WHY IT IS IMPERFECT
---------------------------------
Tracker URLs routinely carry cache-busting or per-impression query strings, so
the same logical request differs between the two loads. Matching therefore
runs in two passes:

  exact       identical URL in the control arm
  path        same scheme://host/path, query ignored; used only when the
              control arm has exactly one such request, so an ambiguous
              many-to-one collapse is never guessed at

Anything else is left unmatched. Pages are also not identical across two loads
minutes apart -- ad slots rotate, content changes -- so treat per-page deltas
as paired samples, not as a controlled diff of one fixed page.

CPU CAVEAT
----------
CPU deltas inherit the sampling noise described in
firefox_crawl_500_tracking.py. If the arms were crawled with several workers,
per-page CPU carries contention and the per-page delta is noisy even when
nothing was blocked; the aggregate is still informative, the individual row
mostly is not.

Usage:
    python src/compare_tracking_arms.py \
        --normal  data/raw/firefox_crawl_500_tracking/normal \
        --private data/raw/firefox_crawl_500_tracking/private \
        --out     data/raw/firefox_crawl_500_tracking
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

# nsresults that mean "the URL classifier stopped this", split by the
# protection responsible. Cryptomining and fingerprinting are active in *both*
# arms (that is ETP Standard), so they must not be counted as savings from
# private browsing.
TRACKING_MARKERS = (
    "NS_ERROR_TRACKING_URI",
    "NS_ERROR_SOCIALTRACKING_URI",
    "NS_ERROR_EMAILTRACKING_URI",
)
OTHER_BLOCK_MARKERS = (
    "NS_ERROR_CRYPTOMINING_URI",
    "NS_ERROR_FINGERPRINTING_URI",
)


def _strip_query(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


def load_pages(arm_dir: Path) -> dict[int, dict]:
    """Per-page records, keyed by crawl index, from the meta sidecars.

    The meta files are read rather than _pages.csv so values keep their types
    and a partially finished (or resumed) arm still loads.
    """
    pages: dict[int, dict] = {}
    for f in sorted(arm_dir.glob("meta_*.json")):
        try:
            with open(f) as fh:
                rec = json.load(fh)
        except Exception:
            continue
        if "idx" in rec:
            pages[rec["idx"]] = rec
    return pages


def load_har_sizes(har_path: Path) -> list[tuple[str, int]]:
    """Every (url, transfer_bytes) observation on one control-arm page.

    Returned as a flat list, not a URL-keyed map, because a page can request
    the same URL more than once (beacons above all) and the blocking arm will
    then report it blocked more than once. Callers claim one observation per
    blocked request so the same bytes are never credited twice.
    """
    obs: list[tuple[str, int]] = []
    if not har_path.exists():
        return obs
    try:
        with open(har_path) as f:
            har = json.load(f)
    except Exception:
        return obs

    for entry in har.get("log", {}).get("entries", []):
        url = entry.get("request", {}).get("url", "")
        if not url:
            continue
        size = entry.get("response", {}).get("_transferSize")
        size = 0 if size is None or size < 0 else int(size)
        obs.append((url, size))
    return obs


def load_blocked(arm_dir: Path, idx: int, slug: str) -> list[dict]:
    f = arm_dir / f"blocked_{idx:04d}_{slug}.json"
    if not f.exists():
        return []
    try:
        with open(f) as fh:
            return json.load(fh).get("blocked", [])
    except Exception:
        return []


def _slug_from_meta(rec: dict, arm_dir: Path) -> str | None:
    """Recover the on-disk slug for a page by globbing its index."""
    hits = list(arm_dir.glob(f"blocked_{rec['idx']:04d}_*.json"))
    if not hits:
        return None
    name = hits[0].name
    return name[len(f"blocked_{rec['idx']:04d}_"):-len(".json")]


PAIRED_COLUMNS = [
    "idx", "url",
    "outcome_normal", "outcome_private",
    "n_requests_normal", "n_requests_private", "n_requests_delta",
    "transfer_bytes_normal", "transfer_bytes_private",
    "bytes_saved", "pct_bytes_saved",
    "cpu_s_normal", "cpu_s_private", "cpu_s_saved",
    "n_blocked_tracking", "n_blocked_other",
    "blocked_observed_bytes", "blocked_matched", "blocked_unmatched",
    "host_cpu_pct_normal", "host_cpu_pct_private",
    "t_start_normal", "t_start_private",
]

BLOCKED_COLUMNS = [
    "page_idx", "page_url", "blocked_url", "resource_type", "failure",
    "protection", "observed_bytes", "match_kind",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal", required=True, help="Control arm directory.")
    ap.add_argument("--private", required=True, help="Blocking arm directory.")
    ap.add_argument("--out", required=True, help="Where to write the tables.")
    args = ap.parse_args()

    normal_dir, private_dir = Path(args.normal), Path(args.private)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pages, p_pages = load_pages(normal_dir), load_pages(private_dir)
    shared = sorted(set(n_pages) & set(p_pages))
    print(f"control arm:  {len(n_pages)} pages")
    print(f"blocking arm: {len(p_pages)} pages")
    print(f"paired:       {len(shared)} pages")

    paired_rows: list[dict] = []
    blocked_rows: list[dict] = []

    for idx in shared:
        n, p = n_pages[idx], p_pages[idx]
        if n.get("url") != p.get("url"):
            print(f"  skipping idx {idx}: URL mismatch between arms")
            continue

        slug = _slug_from_meta(p, private_dir)
        blocked = load_blocked(private_dir, idx, slug) if slug else []

        n_slug = _slug_from_meta(n, normal_dir)
        obs = load_har_sizes(
            normal_dir / f"har_{idx:04d}_{n_slug}.json") if n_slug else []

        observed_total = 0
        n_matched = 0
        n_tracking = n_other = 0

        # Both match passes claim from one shared pool of observations, so a
        # loose path match can never re-credit bytes an exact match already
        # took. Indexes hold positions into `obs`, and `claimed` is the single
        # source of truth for what is still available.
        claimed = [False] * len(obs)
        by_url: dict[str, list[int]] = defaultdict(list)
        by_path_idx: dict[str, list[int]] = defaultdict(list)
        for i, (u, _s) in enumerate(obs):
            by_url[u].append(i)
            by_path_idx[_strip_query(u)].append(i)

        rows_for_page: list[dict] = []
        for b in blocked:
            failure = b.get("failure", "")
            if any(m in failure for m in TRACKING_MARKERS):
                protection, is_tracking = "tracking", True
            elif any(m in failure for m in OTHER_BLOCK_MARKERS):
                protection, is_tracking = "cryptomining_fingerprinting", False
            else:
                protection, is_tracking = "other", False
            n_tracking += is_tracking
            n_other += not is_tracking

            rows_for_page.append({
                "page_idx": idx,
                "page_url": p["url"],
                "blocked_url": b.get("url", ""),
                "resource_type": b.get("resource_type", ""),
                "failure": failure,
                "protection": protection,
                "observed_bytes": "",
                "match_kind": "unmatched",
            })

        def _claim(indices: list[int]) -> int | None:
            for i in indices:
                if not claimed[i]:
                    claimed[i] = True
                    return obs[i][1]
            return None

        # Exact matches claim first; only then may the looser path match take
        # what is genuinely left over.
        for row in rows_for_page:
            size = _claim(by_url.get(row["blocked_url"], []))
            if size is not None:
                row["observed_bytes"] = size
                row["match_kind"] = "exact"
        for row in rows_for_page:
            if row["match_kind"] != "unmatched":
                continue
            # Only path-match where the control arm saw exactly one request
            # for that path; otherwise the right size is a guess among several.
            cands = by_path_idx.get(_strip_query(row["blocked_url"]), [])
            if len(cands) != 1:
                continue
            size = _claim(cands)
            if size is not None:
                row["observed_bytes"] = size
                row["match_kind"] = "path"

        for row in rows_for_page:
            if row["match_kind"] != "unmatched":
                observed_total += row["observed_bytes"]
                n_matched += 1
        blocked_rows.extend(rows_for_page)

        nb, pb = n.get("transfer_bytes", 0), p.get("transfer_bytes", 0)
        ncpu, pcpu = n.get("cpu_total_s"), p.get("cpu_total_s")
        paired_rows.append({
            "idx": idx,
            "url": n["url"],
            "outcome_normal": n.get("outcome"),
            "outcome_private": p.get("outcome"),
            "n_requests_normal": n.get("n_requests", 0),
            "n_requests_private": p.get("n_requests", 0),
            "n_requests_delta": n.get("n_requests", 0) - p.get("n_requests", 0),
            "transfer_bytes_normal": nb,
            "transfer_bytes_private": pb,
            "bytes_saved": nb - pb,
            "pct_bytes_saved": round(100.0 * (nb - pb) / nb, 2) if nb else "",
            "cpu_s_normal": ncpu,
            "cpu_s_private": pcpu,
            "cpu_s_saved": (round(ncpu - pcpu, 3)
                            if ncpu is not None and pcpu is not None else ""),
            "n_blocked_tracking": n_tracking,
            "n_blocked_other": n_other,
            "blocked_observed_bytes": observed_total,
            "blocked_matched": n_matched,
            "blocked_unmatched": len(blocked) - n_matched,
            "host_cpu_pct_normal": n.get("host_cpu_pct_mean"),
            "host_cpu_pct_private": p.get("host_cpu_pct_mean"),
            "t_start_normal": n.get("t_start_iso"),
            "t_start_private": p.get("t_start_iso"),
        })

    with open(out_dir / "paired_pages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIRED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(paired_rows)
    with open(out_dir / "blocked_observed_bytes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLOCKED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(blocked_rows)

    # Restrict headline numbers to pages that loaded cleanly in both arms; a
    # page that timed out on one side has a meaningless delta.
    clean = [r for r in paired_rows
             if r["outcome_normal"].startswith("ok")
             and r["outcome_private"].startswith("ok")]
    saved = [r["bytes_saved"] for r in clean]
    cpu_saved = [r["cpu_s_saved"] for r in clean if r["cpu_s_saved"] != ""]
    matched = [r for r in blocked_rows if r["match_kind"] != "unmatched"]
    tracking_blocked = [r for r in blocked_rows if r["protection"] == "tracking"]

    summary = {
        "n_paired_pages": len(paired_rows),
        "n_clean_pages": len(clean),
        "bytes": {
            "total_normal": sum(r["transfer_bytes_normal"] for r in clean),
            "total_private": sum(r["transfer_bytes_private"] for r in clean),
            "total_saved": sum(saved),
            "median_saved_per_page": statistics.median(saved) if saved else 0,
            "mean_saved_per_page": round(statistics.fmean(saved), 1) if saved else 0,
            "n_pages_negative_saving": sum(1 for s in saved if s < 0),
        },
        "cpu": {
            "total_saved_s": round(sum(cpu_saved), 1) if cpu_saved else None,
            "median_saved_s": (round(statistics.median(cpu_saved), 3)
                               if cpu_saved else None),
            "n_pages": len(cpu_saved),
            "note": "Noisy per page under parallel workers; see module docstring.",
        },
        "blocked_requests": {
            "n_total": len(blocked_rows),
            "n_tracking": len(tracking_blocked),
            "n_cryptomining_fingerprinting": sum(
                1 for r in blocked_rows
                if r["protection"] == "cryptomining_fingerprinting"),
            "n_matched_to_control": len(matched),
            "match_rate": (round(len(matched) / len(blocked_rows), 3)
                           if blocked_rows else None),
            "match_kinds": {
                k: sum(1 for r in blocked_rows if r["match_kind"] == k)
                for k in ("exact", "path", "unmatched")
            },
            "observed_bytes_total": sum(
                r["observed_bytes"] for r in matched if r["observed_bytes"] != ""),
        },
        "caveats": [
            "Cryptomining and fingerprinting blocking is active in BOTH arms "
            "(ETP Standard), so those blocks are not savings from private "
            "browsing; use protection=='tracking' for that.",
            "The two arms are separate page loads, so ad rotation and content "
            "churn contribute to per-page deltas, including negative ones.",
            "Unmatched blocked requests have no observed size; they are "
            "excluded from observed_bytes_total, which is therefore a lower "
            "bound on what was actually saved.",
        ],
    }
    with open(out_dir / "compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    b = summary["bytes"]
    br = summary["blocked_requests"]
    print()
    print("=== Paired comparison ===")
    print(f"Clean paired pages:    {len(clean)}")
    print(f"Transfer, control:     {b['total_normal']/1e6:,.1f} MB")
    print(f"Transfer, blocking:    {b['total_private']/1e6:,.1f} MB")
    print(f"Saved:                 {b['total_saved']/1e6:,.1f} MB "
          f"({100.0*b['total_saved']/b['total_normal']:.1f}%)"
          if b["total_normal"] else "Saved: n/a")
    print(f"Median saved per page: {b['median_saved_per_page']/1e3:,.1f} kB")
    print(f"Pages with a negative delta: {b['n_pages_negative_saving']} "
          f"(ad rotation, not a bug)")
    if summary["cpu"]["total_saved_s"] is not None:
        print(f"CPU saved (sum):       {summary['cpu']['total_saved_s']:,.1f} s "
              f"over {summary['cpu']['n_pages']} pages")
    print()
    print(f"Blocked requests:      {br['n_total']:,} "
          f"({br['n_tracking']:,} tracking, "
          f"{br['n_cryptomining_fingerprinting']:,} crypto/fingerprinting)")
    print(f"Matched to control:    {br['n_matched_to_control']:,} "
          f"({br['match_rate']:.1%})  "
          f"exact={br['match_kinds']['exact']:,} "
          f"path={br['match_kinds']['path']:,} "
          f"unmatched={br['match_kinds']['unmatched']:,}")
    print(f"Observed bytes of blocked requests: "
          f"{br['observed_bytes_total']/1e6:,.1f} MB (lower bound)")
    print()
    print(f"Wrote {out_dir/'paired_pages.csv'}")
    print(f"Wrote {out_dir/'blocked_observed_bytes.csv'}")
    print(f"Wrote {out_dir/'compare_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
