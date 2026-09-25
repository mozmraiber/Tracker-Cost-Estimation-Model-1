"""Run `src/live_ablation.py` over a systematic sample of the top-500 crawl.

WHY A SAMPLE AND NOT THE LIST
-----------------------------
A paired ablation costs two page loads, and a page needs one pair per repeat
per host plus the A/A floor, so the whole 500 is not affordable and most of it
would be wasted anyway: of the 470 pages the ten-pass crawl scored, the
median one has a handful of tracker blocks and many have none, and a page
with no cascading tracker has no cascade to measure. Spending repeats there
buys nothing.

THE SELECTION RULE
------------------
Deterministic, so the sample is reproducible and the manifest records it:

  1. Start from the ten-pass crawl's pooled `paired_pages.csv`. Those pages
     are already known to load, and the crawl already knows what ETP blocked
     on each of them.
  2. Keep pages with at least `--min-blocks` tracking blocks, since that is
     the population `FOLLOWUP_BYTES_PER_REQUEST` is charged over.
  3. Join the hand-labelled page category from
     `data/tranco_500_categories.csv`.
  4. Take the `--per-category` pages with the most tracking blocks in *each*
     category.

Step 4 is the point. `productivity` is 191 of the 470 pages and `news` is 53,
so a sample drawn at random from the list would be mostly productivity sites
and would tell us about the cascade on the pages where there is least of it.
The crawl's own category cuts have already caught two biases the grand total
could not see, and a cascade constant fitted on one category and applied to
all of them is exactly the failure those cuts exist to detect. Ranking within
a category rather than sampling within it is deliberate too: it concentrates
the budget on pages that can actually move the measurement, at the cost of
making the sample the *heavy* end of each category rather than its middle.
That is a real limit on generalising from it and is recorded in the manifest.

WHAT IS STORED
--------------
    <out>/manifest.json   the rule, the parameters, the resolved page list
    <out>/pages/<slug>.json   one full result per page, and the resume unit
    <out>/cells.csv       tidy long: one row per (page, host, rep, width)
    <out>/summary.csv     one row per (page, host, width): n, mean, sd, se

`cells.csv` and `summary.csv` are rebuilt from the per-page JSONs on every
run rather than appended to, so an interrupted run leaves consistent tables
and re-running only costs the pages that have no JSON yet.

USAGE
-----
    python src/run_live_ablation_top500.py --per-category 3 --hosts 3 \\
        --reps 3 --aa-reps 2 --out data/raw/live_ablation_top500

    python src/run_live_ablation_top500.py --rebuild-tables   # no crawling
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from firefox_crawl_500_tracking import build_prefs  # noqa: E402
from live_ablation import WIDTHS, measure_page  # noqa: E402
from replay_ablation import slugify  # noqa: E402

POOLED = (ROOT / "data/raw/firefox_crawl_500_tracking_x10/paired_pages.csv")
CATEGORIES = ROOT / "data/tranco_500_categories.csv"


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0


def load_categories() -> dict[str, str]:
    out = {}
    with open(CATEGORIES) as f:
        for r in csv.DictReader(f):
            out[r["domain"].strip().lower()] = r["category"].strip()
    return out


def page_domain(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0].lstrip("www.").lower()


def select(per_category: int, min_blocks: int) -> list[dict]:
    """The sample, as described in the module docstring."""
    cats = load_categories()
    with open(POOLED) as f:
        rows = list(csv.DictReader(f))
    keep = []
    for r in rows:
        if _f(r, "n_blocked_tracking") < min_blocks:
            continue
        dom = page_domain(r["url"])
        cat = cats.get(dom)
        if not cat:
            continue
        keep.append({"url": r["url"], "domain": dom, "category": cat,
                     "n_blocked_tracking": _f(r, "n_blocked_tracking"),
                     "bytes_saved_tracking": _f(r, "bytes_saved_tracking")})
    by_cat: dict[str, list[dict]] = {}
    for r in keep:
        by_cat.setdefault(r["category"], []).append(r)
    out = []
    for cat in sorted(by_cat):
        ranked = sorted(by_cat[cat],
                        key=lambda r: (-r["n_blocked_tracking"], r["domain"]))
        out.extend(ranked[:per_category])
    return out


# --------------------------------------------------------------------------
# tables


def rebuild_tables(out: Path) -> tuple[int, int]:
    """Re-derive `cells.csv` and `summary.csv` from every page JSON present."""
    cells_p, summ_p = out / "cells.csv", out / "summary.csv"
    cell_rows, summ_rows = [], []
    for jp in sorted((out / "pages").glob("*.json")):
        d = json.loads(jp.read_text())
        if d.get("error"):
            continue
        page, cat = d["url"], d.get("category", "")
        arms = [("aa", "_AA", d.get("aa"))] if d.get("aa") else []
        arms += [("ablation", h, v) for h, v in d.get("hosts", {}).items()]
        if d.get("joint"):
            arms.append(("joint", "_JOINT", d["joint"]))
        for kind, host, block in arms:
            if not block:
                continue
            for rep in block["reps"]:
                for w in WIDTHS:
                    cell_rows.append({
                        "page": page, "category": cat, "host": host,
                        "kind": kind, "rep": rep["rep"],
                        "first_arm": rep["order"][0], "width": w,
                        "control_bytes": rep["control"][w],
                        "ablated_bytes": rep["ablated"][w],
                        "own_bytes": rep["own"],
                        "diff_bytes": (rep["control"][w] - rep["ablated"][w]
                                       - rep["own"]),
                        "n_aborted": rep["n_aborted"],
                        "n_aborted_casc": rep.get("n_aborted_casc", ""),
                    })
            for w in WIDTHS:
                s = block["summary"][w]
                summ_rows.append({
                    "page": page, "category": cat, "host": host, "kind": kind,
                    "width": w, "n": s["n"], "mean_bytes": s["mean"],
                    "sd_bytes": s["sd"], "se_bytes": s["se"],
                    "own_mean_bytes": block.get("own_mean", 0),
                })
    for path, rows in ((cells_p, cell_rows), (summ_p, summ_rows)):
        if not rows:
            continue
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return len(cell_rows), len(summ_rows)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/raw/live_ablation_top500")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--min-blocks", type=int, default=5)
    ap.add_argument("--hosts", type=int, default=3,
                    help="cascading tracker hosts to ablate per page")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--aa-reps", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=45)
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument("--browser-every", type=int, default=5,
                    help="restart Firefox after this many pages")
    ap.add_argument("--joint", action="store_true",
                    help="also ablate every candidate at once, to compare "
                         "the joint counterfactual with the sum of the parts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-measure pages that already have a JSON")
    ap.add_argument("--rebuild-tables", action="store_true",
                    help="rebuild cells.csv/summary.csv and exit")
    a = ap.parse_args()

    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    if a.rebuild_tables:
        n, m = rebuild_tables(a.out)
        print(f"rebuilt {n} cell rows, {m} summary rows")
        return 0

    sample = select(a.per_category, a.min_blocks)
    if a.limit:
        sample = sample[:a.limit]
    manifest = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": ("pooled ten-pass paired_pages.csv; n_blocked_tracking >= "
                 f"{a.min_blocks}; top {a.per_category} per hand-labelled "
                 "page category by n_blocked_tracking"),
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(a).items()},
        "pages": sample,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(sample)} pages selected across "
          f"{len(set(r['category'] for r in sample))} categories")

    done = 0
    pw = browser = None
    try:
        for i, row in enumerate(sample):
            jp = a.out / "pages" / f"{slugify(row['url'])}.json"
            if jp.exists() and not a.force:
                print(f"[{i+1}/{len(sample)}] {row['url']} -- have it")
                continue
            if browser is None or (done and done % a.browser_every == 0):
                if browser is not None:
                    browser.close()
                    pw.stop()
                pw = sync_playwright().start()
                browser = pw.firefox.launch(
                    headless=True,
                    firefox_user_prefs=build_prefs("normal", False))
            print(f"[{i+1}/{len(sample)}] {row['url']} ({row['category']})",
                  flush=True)
            t0 = time.time()
            try:
                res = measure_page(
                    browser, row["url"], hosts=[], auto_top=a.hosts,
                    reps=a.reps, aa_reps=a.aa_reps,
                    timeout_s=a.timeout_s, settle_s=a.settle_s,
                    joint=a.joint)
            except Exception:
                res = {"url": row["url"], "error": traceback.format_exc()}
                print("  FAILED; recorded and moving on")
            res.update({"category": row["category"],
                        "domain": row["domain"],
                        "n_blocked_tracking": row["n_blocked_tracking"],
                        "elapsed_s": round(time.time() - t0, 1)})
            jp.write_text(json.dumps(res, indent=1))
            done += 1
            rebuild_tables(a.out)
    finally:
        if browser is not None:
            browser.close()
            pw.stop()

    n, m = rebuild_tables(a.out)
    print(f"{done} pages measured; {n} cell rows, {m} summary rows in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
