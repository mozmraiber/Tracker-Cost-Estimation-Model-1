"""
Pool a repeated paired tracking crawl into one set of tables.

`compare_tracking_arms.py` joins one control-arm directory to one blocking-arm
directory. This runs it once per repetition of a crawl that was repeated --
`data/raw/firefox_crawl_500_tracking_x10`, ten loads of the Tranco top 500 in
each arm -- and adds the repetitions up, so `tests/top500.py` can read the
result through exactly the code it already had.

WHY REPEAT THE CRAWL AT ALL
---------------------------
Because the denominator, not the estimator, was the limit. Blocking removes
tens of megabytes across the crawl and two loads of the same page routinely
differ by more than that on their own, so a single pass measures the saving to
a factor rather than to a percent. Ten passes divide that noise by the square
root of ten, and -- more usefully -- they *measure* it instead of inferring it.

That second point is the one worth the disk space. A single-pass crawl has no
way to see its own sampling distribution, so `tests/top500.py` estimates it
from the pages ETP never touched: those must shed nothing, so what they shed
is churn. Repetitions give the distribution directly, from how much each rep's
total moves, which means the proxy can be checked for the first time. It
passes: over a single pass it predicts 16.4% for the Disconnect-only delta
where the reps measure 16.2%. `rep_dispersion` writes both into the summary so
the comparison stays visible, and it is the reason the per-category bounds in
`tests/top500.py` can still lean on the proxy at a grain the reps cannot reach.

What the reps buy on top of that is the averaging: ten passes take the same
delta from ±16.2% to ±5.1%, and the whole-page delta from ±74% to ±23.4%. Note
which kinds of noise that removes and which it does not. Repeating a crawl
settles how a *page load* comes out; it says nothing about which *pages* were
crawled, so the per-request bounds and the random-subset bounds, whose
denominator is 271 pages either way, barely move.

WHAT POOLING MEANS HERE
-----------------------
Summed, not averaged, and over a fixed page set.

  * Sums, because the numerator has to be pooled the same way. Every bound in
    `tests/top500.py` is a ratio of an estimate over blocked requests to a
    measured delta, and if the blocked rows are ten reps' worth then the delta
    must be too. A sum keeps every ratio in the file exactly what it was and
    scales the totals by ten, which is the honest description of the data: one
    crawl of 4,700 page loads per arm rather than 470.

  * A fixed page set, because an unbalanced one biases the sums. A page that
    loaded cleanly in six reps and timed out in four would contribute six reps
    of delta against ten reps from its neighbours, so pages that time out often
    -- which are not a random sample of pages -- would be silently
    down-weighted. Only pages clean in *every* rep pair are kept: 470 of the
    500 crawled, with the 30 dropped listed in the summary. Most of those
    failed in one or two reps rather than being broken.

`blocked_observed_bytes.csv` is not aggregated at all: every rep's rows are
concatenated, with a `rep` column added. A blocked request's ground-truth size
is a per-request observation and averaging two of them across reps would throw
away the thing the per-request tests measure, which is the spread.

Usage:
    python src/pool_tracking_reps.py \
        --crawl data/raw/firefox_crawl_500_tracking_x10

which discovers `normal/rep*` and `private/rep*` under it, writes the per-rep
tables to `paired/rep*` and the pooled tables beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

from compare_tracking_arms import (BLOCKED_COLUMNS, PAIRED_COLUMNS, compare)

#: Columns that are extensive: ten reps of a page hold ten reps' worth.
SUMMED = (
    "n_requests_normal", "n_requests_private", "n_requests_delta",
    "transfer_bytes_normal", "transfer_bytes_private", "bytes_saved",
    "tracker_bytes_normal", "tracker_bytes_private", "bytes_saved_tracking",
    "unlisted_tp_bytes_normal", "unlisted_tp_bytes_private",
    "bytes_saved_unlisted_tp",
    "cpu_s_normal", "cpu_s_private", "cpu_s_saved",
    "n_blocked_tracking", "n_blocked_other",
    "blocked_observed_bytes", "blocked_matched", "blocked_unmatched",
)

#: Columns that are intensive: a rate, so pooling means averaging.
AVERAGED = ("host_cpu_pct_normal", "host_cpu_pct_private")

#: The pooled table carries one column the single-crawl one does not.
POOLED_PAIRED_COLUMNS = PAIRED_COLUMNS + ["n_reps"]
POOLED_BLOCKED_COLUMNS = ["rep"] + BLOCKED_COLUMNS


def _num(v, default=0.0):
    """A CSV cell as a float, with blanks and `None` reading as `default`."""
    if v in ("", None):
        return default
    return float(v)


def find_reps(crawl: Path) -> list[str]:
    """Repetition names present in both arms, in order.

    Intersected rather than taken from one arm, so a run that was interrupted
    partway through the second arm pools the reps it actually has in both
    rather than failing or silently pairing rep 7 with rep 8.
    """
    normal = {d.name for d in (crawl / "normal").iterdir() if d.is_dir()}
    private = {d.name for d in (crawl / "private").iterdir() if d.is_dir()}
    return sorted(normal & private)


def clean_everywhere(per_rep: list[list[dict]]) -> tuple[set[int], list[int]]:
    """(pages clean in every rep, pages dropped for not being).

    "Clean" is the same test `tests/top500.py` applies -- both arms reported an
    `ok` outcome -- lifted here so the page set is decided once, at pooling
    time, rather than differently by each caller.
    """
    ok_counts: dict[int, int] = {}
    seen: set[int] = set()
    for rows in per_rep:
        for r in rows:
            idx = int(r["idx"])
            seen.add(idx)
            if (r["outcome_normal"].startswith("ok")
                    and r["outcome_private"].startswith("ok")):
                ok_counts[idx] = ok_counts.get(idx, 0) + 1
    keep = {i for i, n in ok_counts.items() if n == len(per_rep)}
    return keep, sorted(seen - keep)


def pool_pages(per_rep: list[list[dict]], keep: set[int]) -> list[dict]:
    """One row per kept page, summing `SUMMED` and averaging `AVERAGED`."""
    acc: dict[int, dict] = {}
    for rows in per_rep:
        for r in rows:
            idx = int(r["idx"])
            if idx not in keep:
                continue
            out = acc.get(idx)
            if out is None:
                out = acc[idx] = {
                    "idx": idx, "url": r["url"],
                    "outcome_normal": "ok", "outcome_private": "ok",
                    "n_reps": 0,
                    "t_start_normal": r["t_start_normal"],
                    "t_start_private": r["t_start_private"],
                    **{c: 0.0 for c in SUMMED},
                    **{c: 0.0 for c in AVERAGED},
                }
            out["n_reps"] += 1
            for c in SUMMED:
                out[c] += _num(r[c])
            for c in AVERAGED:
                out[c] += _num(r[c])
            # The earliest start across reps, so the column still says when
            # the measurement began rather than when its last rep did.
            for c in ("t_start_normal", "t_start_private"):
                if r[c] and (not out[c] or r[c] < out[c]):
                    out[c] = r[c]

    rows = []
    for out in acc.values():
        for c in AVERAGED:
            out[c] = round(out[c] / out["n_reps"], 2)
        for c in SUMMED:
            # CPU is the only non-integral column, and rounding the rest back
            # to int keeps the file readable as counts and bytes.
            out[c] = (round(out[c], 3) if c.startswith("cpu_s")
                      else int(round(out[c])))
        nb = out["transfer_bytes_normal"]
        out["pct_bytes_saved"] = (round(100.0 * out["bytes_saved"] / nb, 2)
                                  if nb else "")
        rows.append(out)
    return sorted(rows, key=lambda r: r["idx"])


def pool_blocked(per_rep_blocked: list[list[dict]], reps: list[str],
                 keep: set[int]) -> list[dict]:
    """Every rep's blocked rows, concatenated and tagged with their rep."""
    rows = []
    for rep, blocked in zip(reps, per_rep_blocked):
        for r in blocked:
            if int(r["page_idx"]) in keep:
                rows.append({"rep": rep, **r})
    return rows


def rep_dispersion(per_rep: list[list[dict]], keep: set[int]) -> dict:
    """How far each rep's totals sit from the mean of them, and the churn proxy.

    The whole point of repeating the crawl, reduced to four numbers per
    denominator: the mean of the per-rep totals, the standard deviation across
    reps, the standard error of the pooled figure, and -- for the comparison
    the module's bounds used to rest on -- what a single crawl would have
    guessed for that standard error from the pages ETP never touched.

    The churn proxy is computed exactly as `top500.tracking_churn_se_pct` does:
    the per-page standard deviation of the untouched pages' delta, scaled by
    the square root of how many pages were touched. Reported here so the gap
    between the two is a number in a file rather than a claim in a docstring.
    """
    touched = {int(r["idx"]) for rows in per_rep for r in rows
               if int(r["idx"]) in keep and int(r["n_blocked_tracking"]) > 0}
    quiet = keep - touched

    out: dict[str, dict] = {"n_reps": len(per_rep),
                            "n_pages_touched": len(touched),
                            "n_pages_untouched": len(quiet)}
    for name, col in (("bytes_saved", "bytes_saved"),
                      ("bytes_saved_tracking", "bytes_saved_tracking"),
                      ("cpu_s_saved", "cpu_s_saved")):
        totals, churn = [], []
        for rows in per_rep:
            by_idx = {int(r["idx"]): r for r in rows}
            totals.append(sum(_num(by_idx[i][col]) for i in touched
                              if i in by_idx))
            vals = [_num(by_idx[i][col]) for i in quiet if i in by_idx]
            if len(vals) >= 2:
                churn.append(st.stdev(vals) * len(touched) ** 0.5)
        mean = st.fmean(totals)
        sd = st.stdev(totals) if len(totals) >= 2 else float("nan")
        se = sd / len(totals) ** 0.5
        out[name] = {
            "per_rep_mean": round(mean, 1),
            "per_rep_sd": round(sd, 1),
            "pooled_se": round(se, 1),
            "pooled_se_pct": round(100.0 * se / abs(mean), 2) if mean else None,
            "single_rep_sd_pct": round(100.0 * sd / abs(mean), 2) if mean else None,
            "churn_proxy_se_pct": (
                round(100.0 * st.fmean(churn) / abs(mean), 2)
                if churn and mean else None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crawl", required=True,
                    help="Crawl root holding normal/rep* and private/rep*.")
    ap.add_argument("--out", default=None,
                    help="Where to write the pooled tables (default: --crawl).")
    args = ap.parse_args()

    crawl = Path(args.crawl)
    out_dir = Path(args.out) if args.out else crawl
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = find_reps(crawl)
    if not reps:
        print(f"no rep* directories under {crawl}/normal and {crawl}/private")
        return 1
    print(f"pooling {len(reps)} repetitions: {', '.join(reps)}")

    per_rep, per_rep_blocked = [], []
    for rep in reps:
        print(f"\n--- {rep} ---")
        paired, blocked = compare(crawl / "normal" / rep,
                                  crawl / "private" / rep)
        rep_dir = out_dir / "paired" / rep
        rep_dir.mkdir(parents=True, exist_ok=True)
        with open(rep_dir / "paired_pages.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PAIRED_COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(paired)
        with open(rep_dir / "blocked_observed_bytes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BLOCKED_COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(blocked)
        per_rep.append(paired)
        per_rep_blocked.append(blocked)

    keep, dropped = clean_everywhere(per_rep)
    pages = pool_pages(per_rep, keep)
    blocked_rows = pool_blocked(per_rep_blocked, reps, keep)

    with open(out_dir / "paired_pages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOLED_PAIRED_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(pages)
    with open(out_dir / "blocked_observed_bytes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOLED_BLOCKED_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(blocked_rows)

    tracking = [r for r in blocked_rows if r["protection"] == "tracking"]
    matched = [r for r in tracking if r["match_kind"] != "unmatched"]
    summary = {
        "reps": reps,
        "n_pages_pooled": len(pages),
        "n_pages_dropped": len(dropped),
        "dropped_page_idx": dropped,
        "dropped_note": (
            "Pages not clean in every rep pair. Dropping them keeps the page "
            "set balanced across reps; keeping them would down-weight the "
            "pages that time out, which are not a random sample."),
        "blocked_requests": {
            "n_rows": len(blocked_rows),
            "n_tracking": len(tracking),
            "n_tracking_per_rep": round(len(tracking) / len(reps), 1),
            "n_matched_to_control": len(matched),
            "match_rate": (round(len(matched) / len(tracking), 3)
                           if tracking else None),
        },
        "dispersion": rep_dispersion(per_rep, keep),
        "dispersion_note": (
            "`pooled_se_pct` is what repeating the crawl measures directly; "
            "`churn_proxy_se_pct` is what a single crawl would have inferred "
            "from the pages ETP never touched. Where the second is much "
            "smaller, the untouched pages are quieter than the touched ones "
            "and the proxy understates the noise."),
    }
    with open(out_dir / "pool_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== Pooled ===")
    print(f"Pages kept:        {len(pages)} (dropped {len(dropped)} not clean "
          f"in all {len(reps)} reps)")
    print(f"Blocked rows:      {len(blocked_rows):,} "
          f"({len(tracking):,} tracking, "
          f"{summary['blocked_requests']['match_rate']:.1%} matched)")
    d = summary["dispersion"]
    print(f"Touched pages:     {d['n_pages_touched']} "
          f"(untouched {d['n_pages_untouched']})")
    print()
    print(f"{'denominator':22s} {'per rep':>12s} {'1-rep sd':>9s} "
          f"{'pooled se':>10s} {'churn proxy':>12s}")
    for name, scale, unit in (("bytes_saved", 1e6, "MB"),
                              ("bytes_saved_tracking", 1e6, "MB"),
                              ("cpu_s_saved", 1.0, "s ")):
        r = d[name]
        print(f"{name:22s} {r['per_rep_mean']/scale:9.2f} {unit} "
              f"{r['single_rep_sd_pct']:8.1f}% {r['pooled_se_pct']:9.1f}% "
              f"{(r['churn_proxy_se_pct'] or float('nan')):11.1f}%")
    print()
    print(f"Wrote {out_dir/'paired_pages.csv'}")
    print(f"Wrote {out_dir/'blocked_observed_bytes.csv'}")
    print(f"Wrote {out_dir/'pool_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
