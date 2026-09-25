"""Price a subtree and a joint counterfactual at once, by blocking cumulatively.

WHAT THE OTHER TWO INSTRUMENTS EACH MISS
----------------------------------------
The paired crawl blocks a page's whole tracker set and reads the delta. That
is the joint counterfactual -- exactly what ETP does -- but it cannot say
which blocked request the bytes belonged to, and it cannot tell a per-request
subtree from a per-page cost, because a page-level offset divided by k falls
in k exactly as a concave count does. `grade_saturation` concedes this about
itself.

`src/live_ablation.py` blocks one host and leaves the rest of the page
running. That prices a subtree, but it is not the counterfactual the constant
is applied to, and the two disagree by threefold: 14.6 and 14.9 KB a request
over two 125-page runs against the crawl's 47.0. Interaction, the
denominator, and a cross-page intercept have each been tested and none
accounts for it.

THIS INSTRUMENT DOES BOTH
-------------------------
Take the page's ETP-blocked hosts, put them in a random order, and load the
page K+1 times blocking the first k of them for k = 0, 1, ... K. That yields

  * the joint counterfactual, at k = K: every tracker host blocked at once,
    which is the crawl's quantity, on one page;
  * a subtree price per host, as the marginal delta at each step -- and
    priced *in joint context*, with the earlier hosts already gone, rather
    than with the whole ad stack still running to route around the hole;
  * the shape in between, which is the thing neither other instrument can
    see. Fit `cascade(k) = a + b*k` on one page's own sweep and the page is
    its own control: `a` is the cost of blocking anything at all and `b` is
    the cost per additional blocked host. Across pages those two are
    collinear; within a page they are not.

If `a` is small and `b` is near 47 KB, the shipped per-request form is right
and single-host ablation was measuring the wrong thing. If `a` takes most of
the mass, the constant has been charging per request for something that is
mostly per page, and the dashboard over-charges pages with many blocks and
under-charges pages with few.

ORDER IS RANDOMISED AND THAT MATTERS
------------------------------------
The marginal attributed to a host depends on where it falls in the order: a
loader blocked first prunes its whole subtree, the same loader blocked last
prunes nothing because its children went with the earlier blocks. Averaging
over random permutations is what makes the per-host marginals interpretable
rather than an artefact of the sequence, and it is why `--reps` here buys
something different from repeats elsewhere in this repo.

Every arm registers a route, the placebo one included, so interception cost
is identical across k. See `src/live_ablation.py` for why that is not
optional.

USAGE
-----
    python src/sweep_ablation.py --per-category 6 --max-hosts 6 --reps 2 \\
        --out data/raw/sweep_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from firefox_crawl_500_tracking import build_prefs  # noqa: E402
from live_ablation import (PLACEBO_HOST, WIDTHS, load,  # noqa: E402
                           recon)
from replay_ablation import slugify  # noqa: E402
from run_live_ablation_top500 import rebuild_tables as _unused  # noqa: E402,F401
from run_live_ablation_top500 import select  # noqa: E402


def sweep(browser, url: str, hosts: list[str], *, reps: int, timeout_s: int,
          settle_s: float, seed: int = 0) -> dict:
    """Cumulative blocking over `reps` random orders of `hosts`."""
    rng = random.Random(seed)
    runs = []
    for rep in range(reps):
        order = hosts[:]
        rng.shuffle(order)
        steps = []
        base = None
        own_at = {}
        for k in range(len(order) + 1):
            targets = tuple(order[:k]) if k else (PLACEBO_HOST,)
            r = load(browser, url, *targets,
                     timeout_s=timeout_s, settle_s=settle_s)
            if k == 0:
                base = r
                # Own bytes of each host, read off the untouched arm, so the
                # cascade at step k is the delta less what the blocked hosts
                # themselves would have transferred.
                for h in order:
                    own_at[h] = sum(v for hh, v in r["host_bytes"].items()
                                    if hh == h or hh.endswith("." + h))
            own = sum(own_at[h] for h in order[:k])
            steps.append({
                "k": k,
                "blocked": list(order[:k]),
                "bytes": r["bytes"],
                "own": own,
                "n_aborted": r["n_aborted"],
                "n_aborted_casc": r["n_aborted_casc"],
                # The repo's denominator: only the contexts that run code.
                "n_aborted_code": sum(
                    v for k_, v in r["aborted_types"].items()
                    if k_ in ("script", "document")),
                "aborted_types": r["aborted_types"],
                "saved": {w: base["bytes"][w] - r["bytes"][w] for w in WIDTHS},
                "cascade": {w: base["bytes"][w] - r["bytes"][w] - own
                            for w in WIDTHS},
            })
        runs.append({"rep": rep, "order": order, "steps": steps})
        print("    order " + " > ".join(h[:18] for h in order))
        print("      k: " + "  ".join(
            f"{s['k']}:{s['cascade']['third_party']/1e3:+.0f}" for s in steps))
    return {"url": url, "hosts": hosts, "runs": runs}


def rows_from(page: dict) -> list[dict]:
    out = []
    for run in page.get("runs", []):
        for s in run["steps"]:
            for w in WIDTHS:
                out.append({
                    "page": page["url"], "category": page.get("category", ""),
                    "rep": run["rep"], "k": s["k"], "width": w,
                    "n_hosts": len(page["hosts"]),
                    "bytes": s["bytes"][w], "saved": s["saved"][w],
                    "own": s["own"], "cascade": s["cascade"][w],
                    "n_aborted": s["n_aborted"],
                    "n_aborted_casc": s["n_aborted_casc"],
                    "n_aborted_code": s["n_aborted_code"],
                    "last_blocked": s["blocked"][-1] if s["blocked"] else "",
                })
    return out


def rebuild(out: Path) -> int:
    rows = []
    for jp in sorted((out / "pages").glob("*.json")):
        d = json.loads(jp.read_text())
        if d.get("error"):
            continue
        rows.extend(rows_from(d))
    if rows:
        with open(out / "sweep_cells.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/sweep_ablation")
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--min-blocks", type=int, default=5)
    ap.add_argument("--max-hosts", type=int, default=6)
    ap.add_argument("--min-hosts", type=int, default=3,
                    help="skip pages with fewer ETP-blocked hosts than this; "
                         "a sweep needs a slope to fit")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=45)
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument("--browser-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    if a.rebuild:
        print(f"{rebuild(a.out)} rows")
        return 0

    sample = select(a.per_category, a.min_blocks)
    if a.limit:
        sample = sample[:a.limit]
    (a.out / "manifest.json").write_text(json.dumps({
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(a).items()},
        "pages": sample}, indent=1))
    print(f"{len(sample)} pages selected")

    done = 0
    pw = browser = None
    try:
        for i, row in enumerate(sample):
            jp = a.out / "pages" / f"{slugify(row['url'])}.json"
            if jp.exists():
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
                hosts = recon(browser, row["url"], a.max_hosts,
                              timeout_s=a.timeout_s, settle_s=a.settle_s)
                if len(hosts) < a.min_hosts:
                    res = {"url": row["url"], "hosts": hosts, "runs": [],
                           "skipped": f"only {len(hosts)} ETP-blocked hosts"}
                    print(f"  skipped: {res['skipped']}")
                else:
                    res = sweep(browser, row["url"], hosts, reps=a.reps,
                                timeout_s=a.timeout_s, settle_s=a.settle_s)
            except Exception:
                res = {"url": row["url"], "error": traceback.format_exc()}
                print("  FAILED; recorded and moving on")
            res.update({"category": row["category"], "domain": row["domain"],
                        "elapsed_s": round(time.time() - t0, 1)})
            jp.write_text(json.dumps(res, indent=1))
            done += 1
            rebuild(a.out)
    finally:
        if browser is not None:
            browser.close()
            pw.stop()
    print(f"{done} pages; {rebuild(a.out)} rows in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
