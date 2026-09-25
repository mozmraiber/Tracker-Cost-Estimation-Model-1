"""Read what `src/run_live_ablation_top500.py` stored and report the cascade.

The tables are tidy and small, so this does no crawling and re-runs in a
second. Everything it prints is derived from `cells.csv`; nothing is cached.

THREE THINGS IT SPLITS OUT, BECAUSE POOLING THEM HIDES THE ANSWER
-----------------------------------------------------------------
*Width.* The same ablation is accounted over the whole page, over third-party
requests, and over Disconnect-matched ones. Whole-page is unreadable -- its
A/A floor is a median 94 kB a pair and reaches 2.8 MB -- and `listed` is the
width the shipped constant already comes from. `third_party` is the one this
experiment exists to produce: it keeps the unlisted creatives and iframes the
listed width throws away, without the first-party video that drowns the page
width.

*Consent managers.* OneTrust, Osano and Sourcepoint are on the Disconnect
list, so they are picked as ablation candidates, and they are not trackers
with a subtree -- they are gates. Blocking one changes the page's consent
state and therefore every ad decision downstream, in whichever direction that
site's default happens to go: +3.6 MB on braze.com, -634 kB on ted.com. Those
are real effects of blocking and they are not what
`FOLLOWUP_BYTES_PER_REQUEST` models, so they are reported separately rather
than averaged in. See `CMP_HOSTS`.

*The page's own floor.* A cascade is only a measurement if it clears the
spread on *that* page, and the spread ranges over two orders of magnitude
across the sample. A pooled mean that ignores this is mostly reading the
loudest few pages.

The floor is estimated from **every control load on the page**, not from the
A/A arm alone. nist.gov is why. Its two A/A pairs agreed to 0.3 kB, which
would make it the quietest page in the sample -- but the page occasionally
fetches a 4.5 MB asset, in either arm, and two pairs are not enough repeats
to catch a rare event. Pooling the control load from every ablation repeat as
well gives that page eleven observations instead of two and a spread of
megabytes, which is the truth. `control_spread` does this; it costs nothing,
since those loads were run anyway.

Run:

    python src/analyse_live_ablation.py
    python src/analyse_live_ablation.py --out data/raw/live_ablation_top500
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

#: Consent-management platforms. Listed by Disconnect, but ablating one is a
#: change of consent state, not the removal of a subtree. See the docstring.
CMP_HOSTS = (
    "cdn.cookielaw.org",        # OneTrust
    "cmp.osano.com",            # Osano
    "cdn.privacy-mgmt.com",     # Sourcepoint
    "consent.cookiebot.com",
    "cdn.consentmanager.net",
)

WIDTHS = ("page", "third_party", "listed")

#: `third_party` minus `listed`, per observation: the part of a blocked
#: tracker's subtree that lands on hosts no list names. The quantity
#: `FOLLOWUP_BYTES_PER_REQUEST` is a lower bound because it cannot see, and
#: which the paired crawl's own delta cannot measure either -- over the ten
#: passes `bytes_saved_unlisted_tp` carries a 229% pooled standard error.
#: Single-host ablation can, because it never counts the first-party video
#: and third-party CDNs that sink the crawl's version.
UNLISTED = "unlisted"
#: The shipped figure this is measured against. See `FOLLOWUP_BYTES_PER_REQUEST`.
SHIPPED = 47_000


def is_cmp(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in CMP_HOSTS)


def _denom(r: dict) -> float:
    """Cascading aborted requests, falling back to all of them.

    `FOLLOWUP_BYTES_PER_REQUEST` is charged per cascading blocked request, so
    that is the denominator this has to divide by to be comparable with it.
    Runs recorded before `n_aborted_casc` existed have only the wider count
    and read low by whatever share of the ablated host's requests were
    beacons; see the note in `live_ablation.load`.
    """
    v = r.get("n_aborted_casc", "")
    return float(v) if v not in ("", None) and not (isinstance(v, float)
                                                    and math.isnan(v)) \
        else r["n_aborted"]


def load(out: Path) -> tuple[list[dict], list[dict]]:
    def num(rows, keys):
        for r in rows:
            for k in keys:
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", "nan") else math.nan
        return rows
    cells = num(list(csv.DictReader(open(out / "cells.csv"))),
                ("control_bytes", "ablated_bytes", "own_bytes",
                 "diff_bytes", "n_aborted", "n_aborted_casc", "rep"))
    summ = num(list(csv.DictReader(open(out / "summary.csv"))),
               ("n", "mean_bytes", "sd_bytes", "se_bytes", "own_mean_bytes"))
    return cells, summ


def control_spread(cells: list[dict], width: str) -> dict[str, float]:
    """Per page, the sd of the control arm's total over every load of it.

    Every repeat of every arm has a control load -- the placebo -- so a page
    with an A/A pair and three hosts at three repeats has eleven of them. See
    the module docstring for why the A/A arm alone is not enough.
    """
    by = collections.defaultdict(list)
    for r in cells:
        if r["width"] == width:
            by[r["page"]].append(r["control_bytes"])
    return {p: (st.stdev(v) if len(v) > 1 else math.nan)
            for p, v in by.items()}


def per_request(cells: list[dict], width: str, *, drop_cmp: bool) -> tuple:
    """Total cascade over total ablated requests -- the shipped estimator."""
    rows = [r for r in cells
            if r["kind"] == "ablation" and r["width"] == width
            and not (drop_cmp and is_cmp(r["host"]))]
    d = sum(r["diff_bytes"] for r in rows)
    n = sum(_denom(r) for r in rows)
    return d, n, (d / n if n else math.nan)


def bootstrap_pages(cells: list[dict], width: str, *, drop_cmp: bool,
                    iters: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Resample *pages*, the way `fit_followups.py` does.

    The unit of resampling is the page and not the ablation, because two hosts
    ablated on one page share that page's ad stack and are not independent
    draws.
    """
    by = collections.defaultdict(lambda: [0.0, 0.0])
    for r in cells:
        if r["kind"] != "ablation" or r["width"] != width:
            continue
        if drop_cmp and is_cmp(r["host"]):
            continue
        by[r["page"]][0] += r["diff_bytes"]
        by[r["page"]][1] += _denom(r)
    pages = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        s = [by[rng.choice(pages)] for _ in pages]
        n = sum(x[1] for x in s)
        if n:
            draws.append(sum(x[0] for x in s) / n)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def _cells_per_request(cells: list[dict]) -> dict[tuple[str, str], float]:
    """(page, host) -> cascade per ablated request, consent managers dropped."""
    by = collections.defaultdict(lambda: [0.0, 0.0])
    for r in cells:
        if (r["kind"] != "ablation" or r["width"] != "third_party"
                or is_cmp(r["host"])):
            continue
        by[(r["page"], r["host"])][0] += r["diff_bytes"]
        by[(r["page"], r["host"])][1] += _denom(r)
    return {k: v[0] / v[1] for k, v in by.items() if v[1]}


def would_a_table_help(cells: list[dict]) -> None:
    """Grade memorising the cascade, on every key the data can be cut on.

    The question is whether `followup_bytes_for` should look the cascade up
    instead of answering with one number per context, and it is really two
    questions: is there a key on which the cascade is stable, and does the
    browser have that key at block time. This settles the first, which is the
    one that has to be true first.

    Leave-one-out over the (page, host) ablations: predict each cell's cascade
    per request from the *other* cells, once from the grand mean and once from
    the mean of the other cells sharing its key, and compare absolute errors.
    The same shape of test `grade_form` and `grade_shape` use in
    `fit_followups.py`, and for the same reason -- a refinement that cannot
    win a held-out comparison is a refinement that is fitting noise. Cells
    whose key is unique fall back to the grand mean, so a key can only win by
    the cells it actually groups; `grouped` reports how many those are.
    """
    obs = _cells_per_request(cells)
    meta = {}
    for r in cells:
        meta[(r["page"], r["host"])] = (r["page"], r["host"], r["category"])

    keys = {
        "host": lambda k: meta[k][1],
        "page site": lambda k: meta[k][0],
        "page category": lambda k: meta[k][2],
    }
    grand = []
    for k, y in obs.items():
        grand.append(abs(y - st.mean([v for kk, v in obs.items() if kk != k])))
    print(f"  {len(obs)} cells over {len(set(meta[k][1] for k in obs))} hosts "
          f"and {len(set(meta[k][0] for k in obs))} pages")
    print(f"  {'key':16s} {'held-out MAE':>14s} {'grouped':>9s} {'wins':>8s}")
    print(f"  {'one constant':16s} {st.mean(grand)/1e3:11.1f} kB "
          f"{'--':>9s} {'--':>8s}")
    for name, keyf in keys.items():
        errs, grouped, wins = [], 0, 0
        for i, (k, y) in enumerate(obs.items()):
            same = [v for kk, v in obs.items()
                    if kk != k and keyf(kk) == keyf(k)]
            if same:
                e = abs(y - st.mean(same))
                grouped += 1
            else:
                e = grand[i]
            errs.append(e)
            wins += e < grand[i]
        print(f"  {name:16s} {st.mean(errs)/1e3:11.1f} kB {grouped:9d} "
              f"{wins:8d}")

    print("  hosts measured on more than one page, per request:")
    hosts = collections.defaultdict(list)
    for k, v in obs.items():
        hosts[meta[k][1]].append(v)
    for h, v in sorted(((h, v) for h, v in hosts.items() if len(v) > 1),
                       key=lambda kv: -len(kv[1])):
        print(f"    {h:40.40s} " + "  ".join(f"{x/1e3:+8.1f}" for x in v))

    print("  pages with more than one ablation, per request:")
    pages = collections.defaultdict(list)
    for k, v in obs.items():
        pages[meta[k][0]].append(v)
    for pg, v in sorted(((p_, v) for p_, v in pages.items() if len(v) > 1),
                        key=lambda kv: -len(kv[1])):
        print(f"    {pg[:40]:41.41s} " + "  ".join(f"{x/1e3:+8.1f}" for x in v))


def joint_vs_sum(cells: list[dict]) -> None:
    """The joint counterfactual against the sum of the single-host arms.

    This is the instrument's own validity check, and the reason it exists is
    that the wide run reads the listed cascade at 14.6 KB a request where the
    crawl's paired delta reads 47.0. Both claim to measure the same thing, so
    at most one of them does.

    The suspect is interaction. ETP blocks a page's whole ad stack at once;
    single-host ablation removes one member and leaves the rest running, and a
    surviving ad stack re-fetches around the hole -- `cdn.taboola.com` blocked
    on weather.com moves 275 KB onto `static.tblcontent.com`, reproducibly.
    If that generalises, the sum of the parts understates the whole and no
    number of single-host ablations converges on the quantity the constant is
    applied to.

    So: ablate every candidate at once and compare with the sum of ablating
    them one at a time, on the same pages. A ratio near 1 says subtrees are
    separable and single-host ablation is sound. Much above 1 says they are
    not, and the instrument is measuring something narrower than it claims.
    """
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in cells:
        if r["width"] != "third_party" or is_cmp(r["host"]):
            continue
        if r["kind"] == "ablation":
            per[r["page"]]["sum"] += r["diff_bytes"]
            per[r["page"]]["sum_n"] += r["n_aborted"]
        elif r["kind"] == "joint":
            per[r["page"]]["joint"] += r["diff_bytes"]
            per[r["page"]]["joint_n"] += r["n_aborted"]
    rows = [(pg, v) for pg, v in per.items() if v.get("joint_n")]
    if not rows:
        print("  no joint arm in this run")
        return
    # Repeat counts differ between the two arms only if a page failed midway;
    # normalise to per-ablated-request so they are comparable regardless.
    tot_s = sum(v["sum"] for _, v in rows)
    tot_j = sum(v["joint"] for _, v in rows)
    n_s = sum(v["sum_n"] for _, v in rows)
    n_j = sum(v["joint_n"] for _, v in rows)
    print(f"  {len(rows)} pages with both arms")
    print(f"  sum of single-host arms {tot_s/n_s/1e3:+8.1f} kB per request "
          f"({n_s:.0f} ablated requests)")
    print(f"  joint arm               {tot_j/n_j/1e3:+8.1f} kB per request "
          f"({n_j:.0f} ablated requests)")
    if tot_s:
        print(f"  joint / sum             {(tot_j/n_j)/(tot_s/n_s):8.2f}x")
    wins = sum(1 for _, v in rows
               if v["joint_n"] and v["sum_n"]
               and v["joint"] / v["joint_n"] > v["sum"] / v["sum_n"])
    print(f"  joint is larger per request on {wins} of {len(rows)} pages")
    print(f"  {'page':34s} {'sum':>10s} {'joint':>10s}")
    for pg, v in sorted(rows, key=lambda kv: -kv[1]["joint"])[:12]:
        print(f"    {pg[:32]:33.33s} {v['sum']/1e3:+9.1f} "
              f"{v['joint']/1e3:+9.1f} kB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/raw/live_ablation_top500")
    a = ap.parse_args()
    cells, summ = load(a.out)

    pages = sorted(set(r["page"] for r in cells))
    hostcells = set((r["page"], r["host"]) for r in cells
                    if r["kind"] == "ablation")
    print(f"{len(pages)} pages, {len(hostcells)} (page, host) ablations, "
          f"{len(cells)} cells")

    print("\n=== per-page noise floor ===")
    aa = [r for r in summ if r["kind"] == "aa"]
    for w in WIDTHS:
        sds = [r["sd_bytes"] for r in aa
               if r["width"] == w and not math.isnan(r["sd_bytes"])]
        cs = [v for v in control_spread(cells, w).values()
              if not math.isnan(v)]
        # The A/A arm is optional: every ablation repeat runs a control load
        # too, so `control_spread` estimates the floor without it, and a wide
        # run spends the saved pairs on more pages instead.
        aa_txt = (f"A/A pairs: median {st.median(sds)/1e3:7.1f} kB, "
                  f"worst {max(sds)/1e3:9.1f}" if sds
                  else "A/A pairs: not run                       ")
        print(f"  {w:12s} {aa_txt}   |   all control loads: "
              f"median {st.median(cs)/1e3:7.1f} kB, worst {max(cs)/1e3:9.1f}")

    print("\n=== cascade per ablated request ===")
    print(f"  {'width':12s} {'all hosts':>22s} {'consent managers out':>24s}")
    for w in WIDTHS:
        d0, n0, v0 = per_request(cells, w, drop_cmp=False)
        d1, n1, v1 = per_request(cells, w, drop_cmp=True)
        print(f"  {w:12s} {v0/1e3:14.1f} kB (n={n0:.0f}) "
              f"{v1/1e3:16.1f} kB (n={n1:.0f})")

    lo, hi = bootstrap_pages(cells, "third_party", drop_cmp=True)
    _, _, pt = per_request(cells, "third_party", drop_cmp=True)
    print(f"\n  third_party, consent managers out: {pt/1e3:.1f} kB per request"
          f"   95% [{lo/1e3:.1f}, {hi/1e3:.1f}] (bootstrap over pages)")
    print(f"  shipped FOLLOWUP_BYTES_PER_REQUEST = {SHIPPED/1e3:.0f} kB")

    print("\n=== THE UNLISTED HALF: third_party minus listed ===")
    per = collections.defaultdict(dict)
    nab: dict = {}
    for r in cells:
        if r["kind"] != "ablation" or is_cmp(r["host"]):
            continue
        k = (r["page"], r["host"], r["rep"])
        per[k][r["width"]] = r["diff_bytes"]
        nab[k] = _denom(r)
    obs = [(k, v["third_party"] - v["listed"], nab[k]) for k, v in per.items()
           if "third_party" in v and "listed" in v]
    tot = sum(o[1] for o in obs)
    nreq = sum(o[2] for o in obs)
    bypage = collections.defaultdict(lambda: [0.0, 0.0])
    for (pg, _h, _r), d, a in obs:
        bypage[pg][0] += d
        bypage[pg][1] += a
    rng = random.Random(0)
    pages_l = list(bypage)
    draws = []
    for _ in range(4000):
        smp = [bypage[rng.choice(pages_l)] for _ in pages_l]
        k = sum(x[1] for x in smp)
        if k:
            draws.append(sum(x[0] for x in smp) / k)
    draws.sort()
    print(f"  {len(obs)} observations over {len(pages_l)} pages")
    print(f"  unlisted increment {tot/nreq/1e3:+8.1f} kB per request  "
          f"95% [{draws[int(.025*len(draws))]/1e3:+.1f}, "
          f"{draws[int(.975*len(draws))]/1e3:+.1f}] (bootstrap over pages)")
    vals = [v[0] / v[1] for v in bypage.values() if v[1]]
    sd_pg = st.stdev(vals) if len(vals) > 1 else float("nan")
    print(f"  per-page sd {sd_pg/1e3:.1f} kB, so +-10 kB needs about "
          f"{(sd_pg/1e4)**2:.0f} pages at this depth")
    print(f"  for scale, the shipped figure is {SHIPPED/1e3:.0f} kB and the "
          f"crawl cannot measure this at all (229% pooled se)")

    print("\n=== consent managers on their own ===")
    for w in ("third_party",):
        rows = [r for r in cells if r["kind"] == "ablation"
                and r["width"] == w and is_cmp(r["host"])]
        by = collections.defaultdict(float)
        for r in rows:
            by[(r["page"], r["host"])] += r["diff_bytes"] / max(
                len(set(x["rep"] for x in rows if x["page"] == r["page"]
                        and x["host"] == r["host"])), 1)
        for (p, h), v in sorted(by.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {p[:34]:35.35s} {h:24.24s} {v/1e3:+9.1f} kB")

    print("\n=== per page category (third_party, consent managers out) ===")
    cat = collections.defaultdict(lambda: [0.0, 0.0, set()])
    for r in cells:
        if (r["kind"] != "ablation" or r["width"] != "third_party"
                or is_cmp(r["host"])):
            continue
        c = cat[r["category"]]
        c[0] += r["diff_bytes"]
        c[1] += r["n_aborted"]
        c[2].add(r["page"])
    for c in sorted(cat):
        d, n, ps = cat[c]
        print(f"  {c:14s} {d/n/1e3:+9.1f} kB/req  "
              f"({n:.0f} requests, {len(ps)} pages)")

    print("\n=== robust pooled estimates (third_party, consent managers out) ===")
    per_cell = []
    by = collections.defaultdict(lambda: [0.0, 0.0])
    for r in cells:
        if (r["kind"] != "ablation" or r["width"] != "third_party"
                or is_cmp(r["host"])):
            continue
        by[(r["page"], r["host"])][0] += r["diff_bytes"]
        by[(r["page"], r["host"])][1] += r["n_aborted"]
    per_cell = sorted(v[0] / v[1] for v in by.values() if v[1])
    print(f"  median over {len(per_cell)} (page,host) cells "
          f"{st.median(per_cell)/1e3:+8.1f} kB per request")
    k = max(1, len(per_cell) // 10)
    trimmed = per_cell[k:-k]
    print(f"  10% trimmed mean                     "
          f"{st.mean(trimmed)/1e3:+8.1f} kB per request")

    print("\n=== joint counterfactual vs the sum of the parts ===")
    joint_vs_sum(cells)

    print("\n=== would a per-host table beat the single constant? ===")
    would_a_table_help(cells)

    print("\n=== host cells clearing 2 SE, and their page's floor ===")
    floor = control_spread(cells, "third_party")
    rows = [r for r in summ if r["kind"] == "ablation"
            and r["width"] == "third_party" and not math.isnan(r["se_bytes"])]
    sig = [r for r in rows
           if abs(r["mean_bytes"]) > 2 * r["se_bytes"]
           and abs(r["mean_bytes"]) > floor.get(r["page"], 0.0)]
    print(f"  {len(sig)} of {len(rows)} clear both 2 SE and their page's "
          f"control spread")
    for r in sorted(sig, key=lambda r: -r["mean_bytes"]):
        tag = " CMP" if is_cmp(r["host"]) else ""
        print(f"    {r['page'][:28]:29.29s} {r['host']:36.36s} "
              f"{r['mean_bytes']/1e3:+9.1f} +-{r['se_bytes']/1e3:6.1f} kB "
              f"(floor {floor.get(r['page'], float('nan'))/1e3:7.1f}){tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
