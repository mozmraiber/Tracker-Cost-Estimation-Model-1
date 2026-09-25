"""Read the cumulative sweep and split the cascade into per-page and per-host.

The sweep loads a page K+1 times, blocking the first k of its ETP-blocked
hosts in random order, so each page carries its own dose-response curve. That
is what makes the two terms separable here and not in `fit_followups.py`:
across pages a per-page offset and a per-request slope are collinear, but
within one page's sweep the intercept is the jump from k=0 to k=1 and the
slope is everything after it.

Three things are read off it.

  THE SPLIT.  Fit `cascade(k) = a + b*k` on each page's own steps, k >= 1.
  `a` is what blocking anything at all costs the page and `b` what each
  further host costs. Pages are then pooled by summing the fitted mass, so
  the answer is in the same units as `FOLLOWUP_BYTES_PER_REQUEST`.

  THE JOINT TOTAL.  At k = K every host is blocked at once, which is the
  crawl's counterfactual on one page. Divided by the cascading blocks it
  removes, it is directly comparable to the shipped 47 KB -- and to the
  14.9 KB single-host ablation reads, which is the disagreement this
  instrument was built to settle.

  THE MARGINALS.  What each host costs at each position in the order. A
  subtree pruned by an earlier block cannot be pruned again, so a falling
  marginal is nesting and a flat one is independence.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = 47_000


def load_rows(out: Path) -> list[dict]:
    rows = list(csv.DictReader(open(out / "sweep_cells.csv")))
    for r in rows:
        for k in ("rep", "k", "n_hosts", "bytes", "saved", "own", "cascade",
                  "n_aborted", "n_aborted_casc", "n_aborted_code"):
            r[k] = float(r[k]) if r[k] not in ("", None) else math.nan
    return rows


def fit_page(steps: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares `y = a + b*k` over one page's steps, k >= 1."""
    pts = [(k, y) for k, y in steps if k >= 1]
    if len(pts) < 2:
        return math.nan, math.nan
    n = len(pts)
    mk = st.fmean(p[0] for p in pts)
    my = st.fmean(p[1] for p in pts)
    den = sum((p[0] - mk) ** 2 for p in pts)
    if den == 0:
        return math.nan, math.nan
    b = sum((p[0] - mk) * (p[1] - my) for p in pts) / den
    return my - b * mk, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/sweep_ablation")
    ap.add_argument("--width", default="third_party",
                    choices=("page", "third_party", "listed"))
    ap.add_argument("--denominator", default="code",
                    choices=("code", "casc", "all"),
                    help="code = script/document only, which is what "
                         "FOLLOWUP_BYTES_PER_REQUEST is charged over; casc "
                         "also counts xhr/fetch; all counts every abort")
    a = ap.parse_args()
    rows = [r for r in load_rows(a.out) if r["width"] == a.width]
    pages = sorted(set(r["page"] for r in rows))
    print(f"{len(pages)} pages, width={a.width}, "
          f"denominator={a.denominator}")

    # Average the reps at each k, so a page contributes one curve.
    curve: dict[str, dict[float, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    ncasc: dict[str, dict[float, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    dcol = {"code": "n_aborted_code", "casc": "n_aborted_casc",
            "all": "n_aborted"}[a.denominator]
    for r in rows:
        curve[r["page"]][r["k"]].append(r["cascade"])
        n = r.get(dcol)
        ncasc[r["page"]][r["k"]].append(
            r["n_aborted_casc"] if n is None or (isinstance(n, float)
                                                 and math.isnan(n)) else n)

    print("\n=== the split, fitted within each page ===")
    fits = {}
    for p in pages:
        pts = [(k, st.fmean(v)) for k, v in sorted(curve[p].items())]
        fits[p] = fit_page(pts)
    good = {p: v for p, v in fits.items() if not math.isnan(v[0])}
    a_mass = sum(v[0] for v in good.values())
    kmax = {p: max(curve[p]) for p in good}
    b_mass = sum(good[p][1] * kmax[p] for p in good)
    tot = a_mass + b_mass
    print(f"  {len(good)} pages fitted")
    print(f"  per-page term   a: median {st.median(v[0] for v in good.values())/1e3:+8.1f} kB"
          f"   mass {a_mass/1e6:+7.2f} MB ({a_mass/tot:5.0%})")
    print(f"  per-host  term  b: median {st.median(v[1] for v in good.values())/1e3:+8.1f} kB"
          f"   mass {b_mass/1e6:+7.2f} MB ({b_mass/tot:5.0%})")
    rng = random.Random(0)
    ps = list(good)
    draws = []
    for _ in range(4000):
        s = [good[rng.choice(ps)] for _ in ps]
        draws.append(st.fmean(x[1] for x in s))
    draws.sort()
    print(f"  b, bootstrap over pages: mean {st.fmean(v[1] for v in good.values())/1e3:+.1f} kB"
          f"  95% [{draws[int(.025*len(draws))]/1e3:+.1f}, "
          f"{draws[int(.975*len(draws))]/1e3:+.1f}]")

    print("\n=== the joint counterfactual, at k = K ===")
    jc = jn = 0.0
    for p in pages:
        k = max(curve[p])
        jc += st.fmean(curve[p][k])
        jn += st.fmean(ncasc[p][k])
    print(f"  all hosts blocked at once: {jc/1e6:+.2f} MB over {jn:.0f} "
          f"cascading blocks = {jc/jn/1e3:+.1f} kB per request")
    print(f"  shipped constant {SHIPPED/1e3:.0f} kB; single-host ablation "
          f"read 14.9 kB; the crawl reads 47.0 kB")

    print("\n=== marginal per host by position in the order ===")
    bypos = collections.defaultdict(list)
    for p in pages:
        for k in sorted(curve[p]):
            if k >= 1 and (k - 1) in curve[p]:
                bypos[k].append(st.fmean(curve[p][k])
                                - st.fmean(curve[p][k - 1]))
    for k in sorted(bypos):
        v = bypos[k]
        print(f"  step {k:.0f}: median {st.median(v)/1e3:+8.1f} kB   "
              f"mean {st.fmean(v)/1e3:+9.1f} kB   (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
