"""Fit `FOLLOWUP_HOST_SCALE`: a per-host shape for the cascade.

WHY THIS IS SHIPPABLE WHEN THE EARLIER PER-HOST FITS WERE NOT
-------------------------------------------------------------
Two per-site refinements were measured and rejected before this one, and a
third -- a per-host table graded leave-one-out on the first 21-page ablation
run -- lost outright, 253.5 KB against the single constant's 206.6. Three
things were wrong with that grading and all three are fixed here.

  It used the third-party width. That width carries the unlisted ad creatives,
  whose per-impression variation is enormous and is not a property of the host
  at all. The listed width is the one the constant is calibrated on and the one
  that reproduces: across two independent 125-page runs its per-cell
  correlation is 0.79 and the run means agree to 4%.

  It had 48 cells from one run. This pools 646 from four, so most hosts are
  seen more than once and a host mean is a mean rather than a single draw.

  It never checked whether there was between-host signal to find. There is:
  pooled, the between-host variance exceeds the within-host sampling variance,
  which is the condition for a table to be able to help at all.

Graded the same way, leave-one-out over the pooled cells, the table now wins
and wins large: 40.2 KB against 55.7 for the single constant on the mean, and
4.0 KB against 26.7 on the median. Shrinking each host towards the grand mean
was tried with the fitted constant k=0.56 and is *worse* than not shrinking,
on rare hosts (24.4 unshrunk against 30.4 shrunk) as well as common ones, so
the shipped table carries plain host means with a minimum observation count
instead.

WHAT IS SHIPPED IS SHAPE, NOT MAGNITUDE
---------------------------------------
The entries are dimensionless multipliers levelled to 1.0 over the crawl's
mix, exactly as `FOLLOWUP_RUNG_SCALE` is, so the table redistributes the
cascade between hosts without changing what the estimator claims in total.

That is deliberate and it is not timidity. The magnitude is in dispute: the
sweep instrument reads 18.9 KB a request against this crawl's 47.0 on the same
width with the denominators aligned, and 88.2 KB once unlisted bytes are
counted. Shipping absolute per-host bytes would silently pick a side in that
argument on 646 observations. Shipping a levelled shape does not: whatever the
right magnitude turns out to be, the relative ordering of hosts measured here
survives it.

Run:

    python llm-classifier/scripts/fit_host_cascade.py --emit
"""

from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "tests"))

from analyse_live_ablation import is_cmp, load, _denom  # noqa: E402
import top500  # noqa: E402

#: Every ablation run, pooled. They differ in page sample and depth; what they
#: share is the design, so a cell is a cell.
RUNS = ("data/raw/live_ablation_wide", "data/raw/live_ablation_wide2",
        "data/raw/live_ablation_joint", "data/raw/live_ablation_top500")
#: The width the constant is calibrated on, and the only one that reproduces
#: between runs. See the module docstring.
WIDTH = "listed"
#: A host enters the table only with at least this many observations. One
#: observation is a draw from a distribution whose sd is 113 KB.
MIN_OBS = 2
#: Multipliers are clamped, then re-levelled, then clamped again until both
#: hold at once. A host measured at forty times the mean on three pages is
#: telling us something real about those pages and not something to
#: extrapolate to every page it appears on; clamping before levelling would
#: let the levelling push entries back out past the bound.
CLAMP = (0.10, 4.00)

#: Two-label public suffixes seen in this data. Never emitted as entries: an
#: entry for `co.uk` would price every British site's trackers alike.
MULTI_SUFFIX = frozenset((
    "co.uk", "com.br", "com.au", "co.jp", "com.mx", "co.in", "com.tr",
    "co.kr", "com.cn", "co.za", "com.ar", "net.br", "org.uk", "com.sg",
    "com.hk", "co.nz", "com.tw", "ne.jp", "or.jp", "com.pl", "com.ua",
))

#: A host keeps its own entry only if its scale differs from its suffix's by
#: more than this, measured in scale units and so against the average
#: cascade, not against the suffix's own value. Relative was wrong: the two
#: `google-analytics.com` hosts read -2 and +2 KB, a difference of nothing
#: that is infinite in proportion, and it kept an entry for each. Absolute
#: keeps the pair under one `google-analytics.com` entry that also covers
#: every regional shard never crawled.
EXACT_KEEPS_ENTRY = 0.25

#: Hosts whose leftmost label is a per-customer opaque id -- a UUID, or a long
#: hex blob. Permutive and Forter mint one of these per publisher, so the host
#: is real, its measurement is real, and it will never be seen again on any
#: other page. Memorising it costs bytes in the binary and can never pay.
import re as _re  # noqa: E402
OPAQUE = _re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8,}"
    r"|[0-9a-f]{16,}|[0-9a-f]{8}-[0-9a-f]{4,})\.", _re.I)


def pooled_cells() -> dict[tuple[str, str, str], float]:
    obs: dict[tuple[str, str, str], float] = {}
    for run in RUNS:
        d = ROOT / run
        if not (d / "cells.csv").exists():
            continue
        cells, _ = load(d)
        by: dict = collections.defaultdict(lambda: [0.0, 0.0])
        for r in cells:
            if (r["kind"] != "ablation" or r["width"] != WIDTH
                    or is_cmp(r["host"])):
                continue
            k = (run, r["page"], r["host"])
            by[k][0] += r["diff_bytes"]
            by[k][1] += _denom(r)
        obs.update({k: v[0] / v[1] for k, v in by.items() if v[1]})
    return obs


def suffix_of(host: str) -> str:
    """The registrable-ish suffix: last two labels, three past a known
    multi-part public suffix. No PSL dependency, for the reason
    `compare_estimate_vs_etp.is_third_party` gives -- the alternative turns
    `amazon.co.uk` into `co.uk`."""
    parts = host.lower().split(".")
    if len(parts) <= 2:
        return host.lower()
    keep = 3 if ".".join(parts[-2:]) in MULTI_SUFFIX else 2
    return ".".join(parts[-keep:]) if len(parts) > keep else host.lower()


def host_keys(host: str):
    """Every key the lookup will try, most specific first.

    The Rust side walks the same sequence, so a change here is a change
    there. Stops before the bare TLD: two labels is the shortest key, and a
    known multi-part suffix is not a key at all.
    """
    parts = host.lower().split(".")
    for i in range(len(parts) - 1):
        key = ".".join(parts[i:])
        if key.count(".") == 0 or key in MULTI_SUFFIX:
            break
        yield key


def fnv1a32(text: str) -> int:
    """The estimator's `Fnv1a` folded to 32 bits, reproduced exactly."""
    h = 0xcbf29ce484222325
    for b in text.lower().encode():
        h = ((h ^ b) * 0x00000100000001b3) & 0xFFFFFFFFFFFFFFFF
    return ((h >> 32) ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF


def crawl_weights() -> dict[str, int]:
    """Per host, how many of the crawl's blocks actually get charged a cascade.

    The levelling weights, and they have to come from here rather than from
    the ablation runs. `FOLLOWUP_HOST_SCALE` is levelled so that the estimator
    charges the same crawl-wide total as the unmodulated constant -- that is
    what `test_the_cascade_modulation_is_levelled` asserts, and what makes a
    shape safe to ship. Levelling over the ablation sample instead levels over
    the wrong population: it over-weights hosts that happened to be among a
    page's top few by bytes and ignores how often each host is actually
    blocked. Doing that is what first made those tests fail.

    A block counts only if the estimator charges it a cascade at all, which is
    why this goes through `top500.predict` rather than counting blocked
    requests.
    """
    blocks = list(top500.load_all_tracking_blocks())
    direct, _ = top500.predict(blocks)
    total, _ = top500.predict(blocks, include_followups=True)
    out: dict[str, int] = collections.Counter()
    for r, d, t in zip(blocks, direct, total):
        if t > d:
            host = r.url.split("//", 1)[-1].split("/", 1)[0]
            out[host.rsplit("@", 1)[-1].split(":")[0].lower()] += 1
    return out


#: The levelling fixed point, found by iterating `--correct` and folding the
#: result back in: emit, rebuild the extension, measure the residual, repeat.
#: It converges in two rounds and lands at 1.0013, against the 1% that
#: `test_the_cascade_modulation_is_levelled` allows. It is a constant here
#: rather than a step in the pipeline because regenerating the table must not
#: require remembering an argument; pass `--factor 1.0 --correct` to re-derive
#: it after changing the fit.
SHIPPED_FACTOR = 0.9824

#: Quantisation: `scale = q / Q_SCALE`, q a u8. Step 0.0157, which is far
#: finer than anything 646 observations can resolve.
Q_SCALE = 63.75


def quantize(scale: float) -> int:
    return max(0, min(255, round(scale * Q_SCALE)))


def fit() -> tuple[list[tuple[str, float, str]], dict]:
    """Hierarchical entries: a suffix each, plus the hosts that disagree.

    Flat coarsening to the suffix loses -- graded leave-one-out it reads
    45.2 KB against exact-host's 40.2 -- because `doubleclick.net` holds
    `ad` at -58, `googleads` at +0 and `securepubads` at +94 KB, and one
    number for the three is worse than three. Flat exact-host matching
    generalises to nothing: `region1.google-analytics.com` is one of a series
    of regional shards, and `o363271.ingest.sentry.io` is a customer id that
    will never be seen twice.

    Both at once costs almost nothing and fixes both: 41.1 KB held out,
    within noise of exact-host, while covering every shard and customer id
    under a measured suffix. An exact host earns its own entry only when it
    differs from its suffix by more than `EXACT_KEEPS_ENTRY`, which is also
    the compression -- most hosts agree with their suffix and are dropped.
    """
    obs = pooled_cells()
    grand = st.fmean(obs.values())
    by_host: dict[str, list[float]] = collections.defaultdict(list)
    by_suffix: dict[str, list[float]] = collections.defaultdict(list)
    for (_run, _page, host), v in obs.items():
        h = host.lower()
        by_host[h].append(v)
        by_suffix[suffix_of(h)].append(v)

    suffixes = {k: st.fmean(v) / grand for k, v in by_suffix.items()
                if len(v) >= MIN_OBS and k not in MULTI_SUFFIX
                and k.count(".") >= 1}
    exacts = {}
    for h, v in by_host.items():
        if len(v) < MIN_OBS or OPAQUE.match(h) or h in suffixes:
            continue
        m = st.fmean(v) / grand
        base = suffixes.get(suffix_of(h))
        if base is None or abs(m - base) > EXACT_KEEPS_ENTRY:
            exacts[h] = m

    entries = {**suffixes, **exacts}
    kinds = {k: ("suffix" if k in suffixes else "exact") for k in entries}

    def resolve(host: str):
        for key in host_keys(host):
            if key in entries:
                return key
        return None

    cw = crawl_weights()
    weight: dict[str, int] = collections.Counter()
    for host, n in cw.items():
        key = resolve(host)
        if key:
            weight[key] += n
    if not sum(weight.values()):
        for h in entries:
            weight[h] = len(by_host.get(h) or by_suffix.get(h) or [1])

    scale = dict(entries)
    for _ in range(80):
        wsum = sum(weight.get(k, 0) for k in scale)
        if not wsum:
            break
        mean = sum(scale[k] * weight.get(k, 0) for k in scale) / wsum
        scale = {k: min(max(v / mean, CLAMP[0]), CLAMP[1])
                 for k, v in scale.items()}
        if abs(mean - 1.0) < 1e-9:
            break

    rows = sorted(((k, scale[k], kinds[k]) for k in scale),
                  key=lambda r: fnv1a32(r[0]))
    wsum = sum(weight.get(k, 0) for k in scale)
    stats = dict(cells=len(obs), hosts=len(by_host), grand=grand,
                 n_suffix=sum(1 for r in rows if r[2] == "suffix"),
                 n_exact=sum(1 for r in rows if r[2] == "exact"),
                 covered=sum(1 for k in scale if weight.get(k)),
                 weighted=wsum,
                 mean_scale_after=(sum(scale[k] * weight.get(k, 0)
                                       for k in scale) / wsum)
                 if wsum else float("nan"))
    return rows, stats


def emit(rows: list[tuple[str, float, str]]) -> str:
    """Two parallel arrays: sorted FNV-1a/32 keys, and u8 scales.

    Strings are not shipped. A key is the hash of a hostname suffix, so the
    binary carries 5 bytes an entry instead of the hostname, and the lookup
    never compares text. The hostnames are kept in the comments, because a
    table nobody can read is a table nobody can check.
    """
    hashes = ", ".join(f"0x{fnv1a32(k):08x}" for k, _, _ in rows)
    quants = ", ".join(str(quantize(v)) for _, v, _ in rows)
    out = [f"/// FNV-1a/32 of each key, ascending. {len(rows)} entries, "
           f"{len(rows) * 5} bytes with [`FOLLOWUP_HOST_Q`].",
           f"static FOLLOWUP_HOST_KEY: [u32; {len(rows)}] = [",
           _wrap(hashes), "];", "",
           "/// Scale for the key at the same index, as `q / 63.75`.",
           f"static FOLLOWUP_HOST_Q: [u8; {len(rows)}] = [",
           _wrap(quants), "];", "",
           "// The keys above, in the same order, so the table can be read:",
           "//"]
    for k, v, kind in rows:
        star = "*." if kind == "suffix" else "   "
        out.append(f"//   {fnv1a32(k):#010x}  {star}{k:<44s} {v:.2f}")
    return "\n".join(out)


def _wrap(items: str, width: int = 74) -> str:
    line, lines = "   ", []
    for tok in items.split(", "):
        if len(line) + len(tok) + 2 > width:
            lines.append(line.rstrip())
            line = "   "
        line += f" {tok},"
    lines.append(line.rstrip())
    return "\n".join(lines)


def measured_correction() -> float:
    """`flat total / actual total`, read off the estimator as it is built.

    The analytic levelling above gets the weighted mean of the host scales to
    1.0 exactly, and that is still not enough, because it is not the only
    thing multiplying the cascade. `FOLLOWUP_RUNG_SCALE` multiplies it too and
    is levelled over its own margin, so the product of two separately-levelled
    factors is levelled only if they are uncorrelated over the crawl -- and
    they are not, since the hosts with big subtrees are also the ones the
    table places on the specific rungs. Per-request rounding adds a little
    more. Together they left the shipped total 1.41% high against
    `test_the_cascade_modulation_is_levelled`'s 1% bound.

    So the last step is measured rather than derived: price the crawl's blocks
    with the table as built, compare with the flat constant, and fold the
    ratio back in. Run it, rebuild the extension, run it again -- it converges
    immediately, and the test is the check that it did.
    """
    blocks = list(top500.load_all_tracking_blocks())
    direct, _ = top500.predict(blocks)
    total, _ = top500.predict(blocks, include_followups=True)
    cascade = sum(total) - sum(direct)
    n = sum(1 for d, t in zip(direct, total) if t > d)
    flat = top500.FOLLOWUP_BYTES_PER_REQUEST * n
    return flat / cascade if cascade else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true",
                    help="print the Rust array instead of the summary")
    ap.add_argument("--factor", type=float, default=SHIPPED_FACTOR,
                    help="multiply every scale by this before emitting. The "
                         "levelling fixed point: --correct measures the "
                         "residual against the extension as currently built, "
                         "so composing it with the factor already baked into "
                         "that build is the caller's job, and this is how. "
                         "Two rounds have always been enough.")
    ap.add_argument("--correct", action="store_true",
                    help="fold in `measured_correction()`, which needs the "
                         "extension built from the table as it currently "
                         "stands; see that function")
    a = ap.parse_args()
    rows, stats = fit()
    if a.factor is not None:
        rows = [(h, min(max(v * a.factor, CLAMP[0]), CLAMP[1]), k)
                for h, v, k in rows]
        if not a.emit:
            print(f"factor {a.factor:.4f} applied")
    if a.correct:
        c = measured_correction()
        rows = [(h, min(max(v * c, CLAMP[0]), CLAMP[1]), k)
                for h, v, k in rows]
        stats["correction"] = c
        if not a.emit:
            print(f"measured correction {c:.4f} folded in")
    if a.emit:
        print(emit(rows))
        return 0
    print(f"{stats['cells']} cells over {stats['hosts']} hosts")
    print(f"grand mean {stats['grand']/1e3:.1f} kB per request")
    print(f"{stats['n_suffix']} suffix entries + {stats['n_exact']} exact "
          f"overrides = {stats['n_suffix'] + stats['n_exact']} rows, "
          f"{(stats['n_suffix'] + stats['n_exact']) * 5} bytes")
    print(f"{stats['covered']} of them are blocked in the crawl, covering "
          f"{stats['weighted']:,} cascading blocks; weighted mean scale "
          f"after levelling {stats['mean_scale_after']:.4f}")
    lo = sorted(rows, key=lambda r: r[1])
    print("\nlowest:")
    for h, v, k in lo[:6]:
        print(f"  {'*.' if k == 'suffix' else '  '}{h:46.46s} {v:5.2f}")
    print("highest:")
    for h, v, k in lo[-6:]:
        print(f"  {'*.' if k == 'suffix' else '  '}{h:46.46s} {v:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
