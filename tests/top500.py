"""Ground truth from the paired top-500 Firefox crawl, for the estimate tests.

The journey tests measure the estimator against HTTP Archive, which is the
population it was fitted on. This module supplies the other kind of check: a
crawl of the Tranco top 500 where Firefox's own ETP decided what to block, and
the control arm recorded what those same requests really cost.

The crawl is ten passes over the list in each arm, pooled into one set of
tables by `src/pool_tracking_reps.py`, so every total here is ten loads of
each page and every ratio is unaffected by that. `CRAWL_DIR` says what the
repetition changed and what it did not; the short version is that it changed
the page-level denominators a great deal and the per-request and per-subset
bounds hardly at all, because those are limited by which 500 domains were
crawled rather than by how precisely each load was weighed.

Two ground truths come out of it, and they are not equally direct:

  bytes   Per request. The blocking arm names the requests Firefox refused;
          `compare_tracking_arms.py` matched them back to the control arm's
          `_transferSize`. Only the matched ones are usable, so this is the
          60.2% of blocked requests whose URL recurred across the two loads.
          Checked in aggregate, over random subsets of pages, and cut two
          ways that hide different errors: by the kind of *page* a request
          was blocked on (`CATEGORIES`) and by the kind of *tracker* it was
          (`TRACKER_FAMILIES`). The second is the newer and, on this crawl,
          the more revealing: the aggregate reads +2.1% and is the sum of
          six roles contributing between -2.0 and +2.7 points, so nine
          tenths of the estimator's error is cancellation between kinds of
          tracker rather than accuracy.

          Cutting finely is affordable here and nowhere else in this module,
          and the reason is worth carrying: this is the only ground truth
          that is per *request*. 11,290 matched observations support a split
          into six; the page-level denominators below are 271 pages however
          many requests sit on them, which is why the same tracker-role
          split applied to the cascade separates nothing at all (see
          `crawl_roles` in `llm-classifier/scripts/fit_followups.py`).

  CPU     Per page only, because the crawl sampled the Firefox process tree
          rather than attributing cycles to individual requests, and it needs
          a drift correction so large that the corrected figure should be
          read as an order of magnitude. Pages where ETP blocked nothing must
          show no saving; they shed -6.2 s of CPU per page, which over the
          271 pages ETP did touch is worth +1,687 s against a raw delta of
          -890 s. The sign flips on the correction. See `Bounds.cpu_ratio_min`
          for why this crawl is worse at CPU than the single-pass one it
          replaced, and what would fix it.

The page-level byte delta comes in two versions and the difference between
them is the difference between a loud instrument and a quiet one. Over the
whole page it carries every way two loads of the same site can differ; counted
over Disconnect-matched requests alone (`bytes_saved_tracking`, added by
`compare_tracking_arms.py`) most of that drops out. The ten passes put numbers
on it that no single pass could, because each pass is a draw of the same
experiment and the spread across them *is* the sampling distribution: per pass
the whole-page delta has a standard deviation of 74% and the Disconnect-only
one 16.2%, so pooled they carry 23.4% and 5.1%. The tighter page-level bounds
divide by the second, and `Bounds.bytes_tracking_ratio_min` is the sharp one.

The two read as though they must nest, and pooled over all 271 pages they do
not: the Disconnect-only delta counts a subset of the requests the whole-page
delta counts and is *larger*, 559.4 MB against 392.8. That retired the
module's one structural assertion -- `Bounds.bytes_page_ratio_max_raw` records
what replaced it -- on the reading that the blocking arm must be fetching
166.6 MB more of something no list names.

It is not. It is six pages. `page_delta_instability` reads the passes one page
at a time rather than summing them, and finds six of the 271 whose two arms
differ, load to load, by about as much as the page weighs. Leave them out and
the whole-page delta is 542.2 MB against the listed-only 549.7, nested again
to within 1.4% on an instrument carrying 5.1%, and the ratio this module
grades falls from 1.48 to 1.07 -- beside the quiet instrument's 1.05 instead
of 41% above it. The same six move the quiet denominator by -1.7%, which is
the check that this removes churn rather than signal.

`Bounds.bytes_page_ratio_max_stable` is the bound that follows and is the one
page-level statement here that a single-pass crawl could not have made: ten
passes pooled into a total cannot find a page whose loads do not repeat,
because they average it in ten times.

There is also one denominator from outside this crawl entirely:
`data/tranco_500_http_archive.csv`, HTTP Archive's measurement of the same
domains. See `HTTP_ARCHIVE_CSV` for what it is worth and what it is not. It is
the only bound here that does not depend on this crawl being sound, which
matters more than it used to: the cascade constant in `llm-classifier` is now
calibrated against this crawl's quiet denominator, so the bound that divides
by that denominator is a calibration check rather than an independent one.

The tables are produced by:

    python src/firefox_crawl_500_tracking.py --mode normal  --repeats 10 ...
    python src/firefox_crawl_500_tracking.py --mode private --repeats 10 ...
    python src/pool_tracking_reps.py --crawl data/raw/firefox_crawl_500_tracking_x10

and are gitignored, so `available()` reports a skip reason when they are absent.
`data/tranco_500_categories.csv`, which labels the pages, is not: it is a
hand-written file about the Tranco list rather than crawl output.

One thing this crawl still cannot do, stated once here because several
docstrings turn on it: it cannot measure the *unlisted* half of the cascade --
the ad creatives and iframes a blocked tracker would have pulled in from hosts
no list names. Only the whole-page delta could see those, and at 23.4% on a
denominator of 393 MB it cannot separate them from the pages' own churn. Ten
passes were expected to settle this and did the opposite: they moved the
whole-page estimate of the cascade from 70 KB a request to 26 KB, 95%
[-26, 70], which is below the listed-only figure that is a part of it.
`llm-classifier/scripts/fit_followups.py` reproduces all of that, and
`FOLLOWUP_BYTES_PER_REQUEST` ships the listed-only figure as a lower bound.
"""

from __future__ import annotations

import csv
import functools
import json
import random
import statistics as st
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

#: The crawl these tests read: ten loads of the Tranco top 500 in each arm,
#: pooled by `src/pool_tracking_reps.py` into one set of tables in the shape a
#: single pass writes. Every total here is therefore ten passes' worth and
#: every ratio is unchanged by that, which is the point of summing rather than
#: averaging -- see the pooling script for why the page set is fixed too.
#:
#: It replaced a single-pass crawl of the same 500 domains in 2026-09, and the
#: reason was the denominator rather than the estimator. A single pass has no
#: way to see its own sampling distribution: it can only infer it, from the
#: pages ETP never touched, and then has to be believed. Ten passes show it.
#: What they showed is that one pass of this experiment knows the
#: Disconnect-only delta to only ±16.2%. Graded with the estimator as it
#: stood, the sharp bound's ratio ran from 1.07 to 1.72 across the ten
#: passes, so the single pass's 1.47 -- which `bytes_tracking_ratio_*` used
#: to be a ±7% band around -- was one draw from that spread rather than a
#: centre, and the band was narrower than its own sampling noise. Pooling
#: the ten brings the denominator to ±5.1%, measured rather than inferred.
#: (The ratio itself then moved to 1.04, but for a separate reason: the
#: cascade constant was recalibrated on the same pooled crawl. See
#: `Bounds.bytes_tracking_ratio_min`.)
#:
#: Two things about the replacement are worth knowing before reading a number
#: off it. Its passes are *noisier* one for one than the crawl it replaced:
#: that one loaded a page in both arms twelve minutes apart, this one runs the
#: whole control arm before the whole blocking arm, so each pair is about 133
#: minutes apart and carries more of the page's own churn (0.56 MB per page
#: against 0.18 MB). Pooling ten of them more than pays that back. And the
#: churn proxy came out of the comparison well: over a single pass it predicts
#: 16.4% where the passes measure 16.2%, which is the first direct check the
#: repo has had on the method `tracking_churn_se_pct` still uses at grains too
#: fine for the reps to speak to.
CRAWL_DIR = ROOT / "data" / "raw" / "firefox_crawl_500_tracking_x10"
BLOCKED_CSV = CRAWL_DIR / "blocked_observed_bytes.csv"
PAIRED_CSV = CRAWL_DIR / "paired_pages.csv"

#: How many passes `CRAWL_DIR` pools, read off the pooling script's summary.
#: Only the reporting in a few docstrings needs it -- the tables are already
#: pooled -- but `har_pairs` does, and so does anything that wants a per-pass
#: figure out of a ten-pass total.
POOL_SUMMARY = CRAWL_DIR / "pool_summary.json"

# `context_for` maps Playwright's resource types onto the table's contexts.
# Imported rather than restated so the test cannot drift from the analysis.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def available() -> str | None:
    """Skip reason if the crawl output is not on disk, else None."""
    missing = [p for p in (BLOCKED_CSV, PAIRED_CSV) if not p.exists()]
    if not missing:
        return None
    names = ", ".join(str(p.relative_to(ROOT)) for p in missing)
    return (f"{names} not present (gitignored); regenerate with "
            f"src/firefox_crawl_500_tracking.py --repeats 10 and "
            f"src/pool_tracking_reps.py to run the top-500 estimate tests")


@functools.cache
def reps() -> tuple[str, ...]:
    """The passes `CRAWL_DIR` pools, in order, or empty if it is not pooled.

    Empty rather than an error for a single-pass crawl, so anything that only
    wants to report "over N passes" degrades to saying nothing instead of
    failing on a directory this module can otherwise read fine.
    """
    if not POOL_SUMMARY.exists():
        return ()
    with open(POOL_SUMMARY) as f:
        return tuple(json.load(f).get("reps", ()))


@functools.cache
def n_passes() -> int:
    """How many loads of each page the totals in these tables add up.

    Every total this module reports is summed over the crawl's passes, which
    is what keeps its ratios readable: numerator and denominator are pooled
    the same way and the factor cancels. It does not cancel in the one
    comparison whose denominator comes from outside -- HTTP Archive measured
    each domain once -- so that test divides by this and nothing else does.

    One for a crawl that was not repeated, so the division is a no-op there.
    """
    return len(reps()) or 1


def har_pairs(page_idx: int) -> list[tuple[Path, Path]]:
    """(control HAR, blocking HAR) for one page, one pair per pass.

    The one place that knows the crawl is a directory of passes rather than a
    single pair of arm directories. `llm-classifier/scripts/fit_followups.py`
    reads the HARs directly to measure the cascade a second way, from request
    sets rather than byte totals, and it should not have to learn the layout.
    """
    out = []
    for rep in reps() or ("",):
        normal = sorted((CRAWL_DIR / "normal" / rep).glob(f"har_{page_idx:04d}_*.json"))
        private = sorted((CRAWL_DIR / "private" / rep).glob(f"har_{page_idx:04d}_*.json"))
        if normal and private:
            out.append((normal[0], private[0]))
    return out


#: Instability above which a page's whole-page byte delta is read as its own
#: churn rather than as a measurement of blocking, and the page is left out of
#: the whole-page denominator. A page scores its per-pass standard deviation
#: as a share of its own size, so 1.0 means "the two arms differ, load to
#: load, by about as much as the page weighs".
#:
#: That is an extreme bar and it is meant to be. The crawl's touched pages
#: score 0.026 at the median and 0.48 at the 95th percentile, so 1.0 is twice
#: the 95th percentile and cuts between it and the 99th at 1.8, removing six
#: of 271. What it removes is nevertheless most of the whole-page instrument's
#: noise -- see `page_delta_instability` for the numbers, which are the reason
#: this exists rather than an argument for it.
#:
#: It is a threshold on scale rather than on rank because the distribution has
#: that shape: a bulk of pages whose loads repeat to within a few percent, and
#: a handful that do not repeat at all. Cutting by quantile would make which
#: pages go depend on how many quiet ones happened to be crawled beside them.
#:
#: 1.0 rather than some other scale because the answer is flat around it and
#: not beyond it. From 0.5 to 2.0 the whole-page ratio runs 1.05 to 1.15 --
#: within 8% of the 1.07 at 1.0, on page sets of 259 to 269 -- so the choice
#: inside that range is not load-bearing. Outside it is: at 3.0 only
#: nytimes.com is left in and the ratio is back to 1.45, and below 0.4 the
#: filter starts discarding real pages, 13% of the predicted bytes by 0.3.
MAX_PAGE_INSTABILITY: float = 1.0


@functools.cache
def page_delta_instability() -> dict[int, float]:
    """Per page, how far its whole-page byte delta moves from pass to pass.

    The one thing here that a single-pass crawl cannot compute, and the reason
    the pooled tables keep their per-pass parts. A page's score is the standard
    deviation of its ten per-pass whole-page deltas over its own control-arm
    transfer size, so it is dimensionless, independent of how many passes were
    run, and says how much of what the two arms differ by on that page is the
    page rather than the blocking.

    WHAT IT IS FOR. The whole-page delta is this module's loud instrument:
    392.8 MB pooled with a standard error of 23.4%, against 559.4 MB and 5.1%
    for the Disconnect-only one. Six pages out of 271 are why. They score 1.0
    to 3.9 here -- tradingview.com, a streaming charting app, at 2.4;
    nytimes.com at 3.9 -- and between them they carry -149.4 MB of delta
    against 3.9 MB of predicted saving, which is 0.7% of the estimate setting
    38% of the denominator.

    WHY THE CUT IS NOT CIRCULAR, which matters because dropping pages until a
    ratio behaves is exactly what this would otherwise be. The score is
    computed from the spread of a page's deltas and its size. It does not read
    the level of the delta, the estimator, or the ratio, so a page cannot be
    cut for disagreeing. And the crawl has a second instrument to check the
    cut against, one that was never noisy: over the same six pages the
    Disconnect-only delta moves by -1.7%, from 559.4 MB to 549.7 MB, while the
    whole-page delta moves by +38%, from 392.8 MB to 542.2 MB. A filter that
    removed signal would take both down together.

    WHAT IT SETTLES. That the module's standing puzzle was those six pages and
    not a property of blocking. A strict subset of the whole-page delta --
    the part of it on Disconnect-matched requests -- measured 42% *larger*
    than the whole, which is impossible unless the blocking arm fetches more
    of something no list names, and `Bounds.bytes_page_ratio_max_raw` retired
    the module's one structural assertion over it. On the stable 265 the two
    come back in order to within their own noise, 542.2 MB whole against
    549.7 MB listed-only, a 1.4% gap against the 5.1% the quiet instrument
    carries. The 166.6 MB of unlisted fetching at t = -2.1 was tradingview.com
    and five others.

    Empty for a crawl whose per-pass tables are absent, which reads through
    `unstable_pages` as "no page is unstable". A single-pass crawl genuinely
    cannot tell, so the bound that needs this skips on an empty result rather
    than grading an unfiltered denominator against a filtered band; the
    unfiltered bound beside it still runs.
    """
    out: dict[int, float] = {}
    per_rep = [CRAWL_DIR / "paired" / rep / "paired_pages.csv" for rep in reps()]
    if len(per_rep) < 2 or not all(p.exists() for p in per_rep):
        return out

    deltas: dict[int, list[float]] = {}
    sizes: dict[int, list[float]] = {}
    for path in per_rep:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                idx = int(row["idx"])
                deltas.setdefault(idx, []).append(float(row["bytes_saved"]))
                sizes.setdefault(idx, []).append(
                    float(row["transfer_bytes_normal"]))
    for idx, per_pass in deltas.items():
        size = st.fmean(sizes[idx])
        if len(per_pass) < 2 or size <= 0:
            continue
        out[idx] = st.stdev(per_pass) / size
    return out


@functools.cache
def unstable_pages() -> frozenset[int]:
    """Page indices the whole-page delta cannot measure. See above."""
    scores = page_delta_instability()
    return frozenset(idx for idx, score in scores.items()
                     if score > MAX_PAGE_INSTABILITY)


@dataclass(frozen=True)
class BlockedRequest:
    """One request Firefox refused, with what it really cost."""

    url: str
    resource_type: str
    observed_bytes: int
    page_idx: int
    page_url: str
    category: str


def _blocked_rows() -> list[dict]:
    """Every tracking-protection block in the crawl, as raw CSV rows.

    Cryptomining and fingerprinting blocks are excluded: the estimator models
    tracking content, and those are a different protection and a handful of
    requests. The control arm blocks nothing at all, so they are real savings
    -- `compare_tracking_arms.py` reports them under their own protection
    rather than folding them in, and nothing here asks the estimator about them.
    """
    with open(BLOCKED_CSV, newline="") as f:
        return [r for r in csv.DictReader(f) if r["protection"] == "tracking"]


def _as_request(row: dict) -> BlockedRequest:
    raw = row["observed_bytes"]
    return BlockedRequest(
        url=row["blocked_url"],
        resource_type=row["resource_type"],
        observed_bytes=int(raw) if raw not in ("", None) else -1,
        page_idx=int(row["page_idx"]),
        page_url=row["page_url"],
        category=category_for(row["page_url"]),
    )


def load_blocked_requests() -> list[BlockedRequest]:
    """Tracking-protection blocks that could be priced against the control arm.

    This covers 11,290 of the 18,749 tracking blocks. The other 7,459 have no
    control-arm twin, so there is no per-request number to compare a prediction
    against -- the match *is* the measurement, and dropping the filter would
    only turn "unknown" into a spurious zero. They are not left ungraded,
    though: `test_blocked_bytes_do_not_exceed_page_level_saving` prices all
    18,749 against `MeasuredPageCost.bytes_saved`, which is measured at page
    level and so covers the unmatched ones too. That is a one-sided bound
    rather than a bias check, which is the most the data supports.

    `observed_bytes` is the load-bearing test: a row has a priced twin or it
    does not. `match_kind == "unmatched"` names exactly the same rows --
    `compare_tracking_arms.py` writes the two together -- so it is not
    filtered on separately.
    """
    return [_as_request(r) for r in _blocked_rows() if r["observed_bytes"] != ""]


def load_all_tracking_blocks() -> list[BlockedRequest]:
    """Every tracking-protection block, whether or not it could be priced.

    The per-request byte checks need a control-arm observation to compare
    against, so they use `load_blocked_requests`. A page-level check needs no
    such match -- the estimator prices a URL from the URL -- and the saving it
    is compared against covers *all* the blocking, so restricting to the
    matched 60.2% here would compare a part against the whole.
    `observed_bytes` is -1 for the unpriced ones, which no caller should read.
    """
    return [_as_request(r) for r in _blocked_rows()]


def host_of(url: str) -> str:
    """The host a URL was fetched from, lower-cased and without the port.

    The grain `CascadeRoot` is keyed on: two blocked requests share a pruned
    subtree when they share a host. Registrable domain is the arguable
    alternative -- `securepubads.g.doubleclick.net` and
    `stats.g.doubleclick.net` are one vendor -- and it was tried and is
    worse: it merges 5,441 host-firsts into 5,167 and loses to this grain on
    per-page error, winning 28% of paired bootstraps. Vendors run their
    loaders and their beacon endpoints on separate hostnames and the subtrees
    are separate too. `host_dedup` in
    `llm-classifier/scripts/fit_followups.py` re-measures both.
    """
    return urlparse(url).netloc.split(":")[0].lower()


@functools.cache
def blocked_by_pass() -> tuple[tuple[BlockedRequest, ...], ...] | None:
    """Every tracking block, one tuple per pass, in the order the crawl saw it.

    `load_all_tracking_blocks` reads the pooled table, which is the ten passes
    concatenated with no column saying which is which. That is the right shape
    for everything that divides one total by another, and the wrong shape for
    anything that asks what a *page load* looked like: a host blocked once per
    pass appears ten times there, so deduplicating by host over the pooled
    table would collapse the passes as well as the within-page repeats. On
    this crawl that is the difference between a 23% reduction and a 92% one.

    None for a crawl whose per-pass tables are absent, which the one caller
    reads as a skip. See `page_delta_instability`, which needs the passes kept
    apart for the same reason and says more about why.
    """
    paths = [CRAWL_DIR / "paired" / rep / "blocked_observed_bytes.csv"
             for rep in reps()]
    if len(paths) < 2 or not all(p.exists() for p in paths):
        return None
    out = []
    for path in paths:
        with open(path, newline="") as f:
            out.append(tuple(_as_request(r) for r in csv.DictReader(f)
                             if r["protection"] == "tracking"))
    return tuple(out)


def load_blocked_by_page() -> dict[int, list[BlockedRequest]]:
    """The priced requests, grouped by the page they were blocked on."""
    pages: dict[int, list[BlockedRequest]] = {}
    for r in load_blocked_requests():
        pages.setdefault(r.page_idx, []).append(r)
    return pages


def load_blocked_by_category() -> dict[str, list[BlockedRequest]]:
    """The priced requests, grouped by the category of the page they are on."""
    out: dict[str, list[BlockedRequest]] = {k: [] for k in CATEGORIES}
    for r in load_blocked_requests():
        out[r.category].append(r)
    return out


@dataclass(frozen=True)
class CrawledPage:
    """One paired page: what the two arms did on it."""

    idx: int
    url: str
    category: str
    n_blocked_tracking: int
    bytes_saved: int
    bytes_saved_tracking: int
    #: Third-party bytes the Disconnect list does *not* name. The part of a
    #: blocked tracker's subtree that `bytes_saved_tracking` cannot see, and
    #: the reason the shipped cascade is documented as a lower bound.
    bytes_saved_unlisted_tp: int
    cpu_s_saved: float | None


@functools.cache
def load_pages() -> tuple[CrawledPage, ...]:
    """Every page both arms loaded successfully.

    Kept as the one place the outcome filter lives, so a subset test and the
    whole-crawl test cannot disagree about which pages exist.
    """
    with open(PAIRED_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["outcome_normal"].startswith("ok")
                and r["outcome_private"].startswith("ok")]
    return tuple(CrawledPage(
        idx=int(r["idx"]),
        url=r["url"],
        category=category_for(r["url"]),
        n_blocked_tracking=int(r["n_blocked_tracking"]),
        bytes_saved=int(r["bytes_saved"]),
        bytes_saved_tracking=int(r.get("bytes_saved_tracking") or 0),
        bytes_saved_unlisted_tp=int(r.get("bytes_saved_unlisted_tp") or 0),
        cpu_s_saved=(float(r["cpu_s_saved"])
                     if r["cpu_s_saved"] not in ("", None) else None),
    ) for r in rows)


@dataclass(frozen=True)
class MeasuredPageCost:
    """What blocking measurably saved, in aggregate, across the crawl.

    `cpu_saved_s` and `bytes_saved` are drift-corrected: the pages ETP never
    touched are used to estimate crawl-to-crawl churn, and that estimate is
    subtracted from the pages it did touch.

    Both corrections run negative on this crawl, so both *raise* the figures
    they correct, and the byte one is modest where the CPU one is not. Byte
    churn is -155 kB per untouched page over ten passes, which takes the raw
    392.8 MB on the pages ETP touched to 435.0 MB corrected. CPU churn is
    -6.2 s per untouched page, which takes a raw delta of -890 s to +797 s:
    a correction twice the size of its own result, and the reason
    `Bounds.cpu_ratio_min` says to read the CPU band sceptically. The
    single-pass crawl this replaced had both drifts positive, so neither the
    sign nor the size is a property of the method.

    `bytes_saved_raw` is the same page-level byte delta with nothing subtracted,
    kept so a bound can be stated against what was literally measured rather
    than only against a figure this module derived.

    `bytes_saved_tracking` is the quieter instrument, and the one the sharper
    bounds divide by. It is the same two-arm delta counted only over requests
    the Disconnect list names, on either side, so a page's first-party media,
    carousels and lazy images -- which is where a repeated page load differs
    -- drop out of it entirely. Across the ten passes its total has a
    standard deviation of 16.2% against the whole page's 74%, so pooled it is
    known to 5.1% against 23.4%. `compare_tracking_arms.py` computes it; the
    cost is that it cannot see what a blocked tracker would have pulled in
    from a host the list does not name.

    It did not, however, sit below the whole-page figure the way that cost
    implies it should: 559.4 MB raw against the whole page's 392.8 MB, over a
    strict subset of the same requests. That inversion turned out to be six
    pages rather than a fact about blocking, and the `_stable` fields are what
    is left when they are gone: 542.2 MB whole-page against 549.7 MB
    listed-only, back in order to within 1.4% on a quiet instrument that
    carries 5.1%. `page_delta_instability` picks the six out, from the spread
    of each page's per-pass deltas and nothing else, and the case that it is
    measuring churn rather than removing signal is that the same six move the
    quiet denominator by -1.7% and the loud one by +38%.

    So the stable whole-page delta is the better instrument on both counts and
    the bounds that divide by it are the tighter ones: its standard error is
    16.2% against the full set's 23.4%, and it is a ratio of 1.07 rather than
    1.48. `bytes_saved_raw` is kept beside it unfiltered, because a bound
    stated against every page the crawl touched is the one that needs no
    argument. See `Bounds.bytes_page_ratio_max_stable`.

    Every field here is still noisy at the scale of the effect, which is why
    the page-level bounds in `Bounds` are wide and the subset tests ask about
    stability rather than accuracy. What changed with the ten passes is that
    the noise is measured rather than inferred -- and that the cascade
    constant in `llm-classifier` *is* now fitted against
    `bytes_saved_tracking`, which costs the bound that divides by it its
    independence.
    """

    n_pages_with_blocks: int
    n_pages_without_blocks: int
    n_pages_unstable: int
    cpu_saved_s: float
    bytes_saved: float
    bytes_saved_raw: float
    bytes_saved_raw_stable: float
    bytes_saved_tracking: float
    bytes_saved_tracking_raw: float
    bytes_saved_tracking_raw_stable: float
    bytes_saved_unlisted_tp: float
    bytes_saved_unlisted_tp_raw: float
    cpu_drift_per_page_s: float


def measure(pages: Sequence[CrawledPage],
            untouched: Sequence[CrawledPage] | None = None) -> MeasuredPageCost:
    """Aggregate the measured saving over `pages`.

    `untouched` supplies the churn estimate and defaults to every page in the
    crawl that ETP left alone -- the whole set, not the part of it that happens
    to fall in a subset, because drift is a property of the crawl rather than
    of the pages a caller picked. A subset test therefore measures its own
    saving against the crawl's drift, which is what makes subsets comparable
    to each other and to the whole.
    """
    touched = [p for p in pages if p.n_blocked_tracking > 0]
    quiet = list(untouched if untouched is not None
                 else [p for p in load_pages() if p.n_blocked_tracking == 0])

    cpu_touched = [p.cpu_s_saved for p in touched if p.cpu_s_saved is not None]
    cpu_quiet = [p.cpu_s_saved for p in quiet if p.cpu_s_saved is not None]
    cpu_drift = st.fmean(cpu_quiet) if cpu_quiet else 0.0
    bytes_drift = st.fmean([p.bytes_saved for p in quiet]) if quiet else 0.0
    bytes_touched = sum(p.bytes_saved for p in touched)
    # The tracking delta gets its own drift, from the same pages: restricting
    # to Disconnect-matched requests removes most of the churn as well as most
    # of the signal, so the whole page's correction would be four times too
    # large here.
    track_drift = (st.fmean([p.bytes_saved_tracking for p in quiet])
                   if quiet else 0.0)
    track_touched = sum(p.bytes_saved_tracking for p in touched)
    # And its own again for the unlisted third-party delta, for the same
    # reason: it is a third width of the same difference, so it carries a
    # third amount of churn and cannot borrow either of the other two.
    unl_drift = (st.fmean([p.bytes_saved_unlisted_tp for p in quiet])
                 if quiet else 0.0)
    unl_touched = sum(p.bytes_saved_unlisted_tp for p in touched)

    # The same two raw deltas over the pages whose loads repeat. The filter is
    # applied here rather than by the caller so a subset test gets it too, and
    # applied to both deltas rather than only the loud one so the pair stays
    # comparable -- the point of the `_stable` figures is that they are the
    # same difference over the same pages, counted two ways.
    unstable = unstable_pages()
    stable = [p for p in touched if p.idx not in unstable]

    return MeasuredPageCost(
        n_pages_with_blocks=len(touched),
        n_pages_without_blocks=len(quiet),
        n_pages_unstable=len(touched) - len(stable),
        cpu_saved_s=sum(cpu_touched) - cpu_drift * len(cpu_touched),
        bytes_saved=bytes_touched - bytes_drift * len(touched),
        bytes_saved_raw=bytes_touched,
        bytes_saved_raw_stable=sum(p.bytes_saved for p in stable),
        bytes_saved_tracking=track_touched - track_drift * len(touched),
        bytes_saved_tracking_raw=track_touched,
        bytes_saved_tracking_raw_stable=sum(
            p.bytes_saved_tracking for p in stable),
        bytes_saved_unlisted_tp=unl_touched - unl_drift * len(touched),
        bytes_saved_unlisted_tp_raw=unl_touched,
        cpu_drift_per_page_s=cpu_drift,
    )


def load_measured_page_cost() -> MeasuredPageCost:
    """The whole crawl's measured saving."""
    return measure(load_pages())


def tracking_churn_se_pct(pages: Sequence[CrawledPage]) -> float:
    """How much of `pages`' Disconnect-only delta is this crawl's own churn.

    The pages ETP left alone must shed nothing, so what they do shed is the
    crawl's load-to-load variance. Its standard deviation per page, scaled up
    by the square root of how many pages are in `pages`, is what a re-crawl
    would be expected to move their delta by; this returns that as a
    percentage of the delta itself.

    It is what decides at which grain the quiet instrument can be used at all.
    Over the whole crawl it is 5.2%, against 5.1% measured directly from the
    spread across the crawl's ten passes -- the first check this repo has had
    on the method, and it came out well. Per pass the proxy says 16.4% where
    the passes say 16.2%. That agreement is what licenses using it per
    category, where ten passes over the same 271 pages have nothing to say:
    it runs from 5.6% on news to 48.7% on shopping, so only some categories
    can be bounded. See `Bounds.gradeable_tracking_se_pct`. The same figure
    on the whole-page delta is 14.9% over the crawl and into the hundreds per
    category, which is why none of this works on that denominator.

    Returns `inf` for a group whose delta is zero or negative, which the
    callers read as "not gradeable" rather than as a division by zero.
    """
    touched = [p for p in pages if p.n_blocked_tracking > 0]
    quiet = [p for p in load_pages() if p.n_blocked_tracking == 0]
    delta = sum(p.bytes_saved_tracking for p in touched)
    if delta <= 0 or len(quiet) < 2 or not touched:
        return float("inf")
    sd = st.stdev([p.bytes_saved_tracking for p in quiet])
    return 100.0 * sd * len(touched) ** 0.5 / delta


def random_page_subsets(fraction: float = 0.5, n: int = 200,
                        seed: int = 20260922) -> list[list[CrawledPage]]:
    """`n` random subsets of the pages ETP acted on, seeded so they never move.

    Sampling pages rather than requests on purpose: a page is the unit the
    measured saving is recorded at, and the requests on one page are anything
    but independent of each other. Without replacement within a subset, so a
    subset is a smaller crawl rather than a bootstrap resample.
    """
    touched = [p for p in load_pages() if p.n_blocked_tracking > 0]
    k = max(1, int(round(fraction * len(touched))))
    rng = random.Random(seed)
    return [rng.sample(touched, k) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Page categories
# --------------------------------------------------------------------------- #
#: What kind of page each crawled domain is, from `CATEGORY_FILE`.
#:
#: The aggregate byte bound is a sum over all 500 domains, so a bias confined
#: to one kind of page cancels against the rest and passes. That is what these
#: exist to catch, and they have twice: news and media pages once ran -18.7%
#: against the rest of the crawl's +0.1% while the aggregate sat at -7.1%,
#: comfortably inside the bound (the host rungs in `llm-classifier` closed it;
#: see `Level::Host`), and `infrastructure` runs +33.7% today while the
#: aggregate reads +2.1%.
#:
#: This replaces an earlier two-way news/other split. Seven categories cost
#: nothing extra to carry and say more: the old split put a bank, a CDN root
#: and a streaming site in one bucket called `other`, which is where the
#: +33.7% was hiding.
#:
#: Membership was assigned from the domain name alone, on editorial grounds,
#: and deliberately not from prediction error -- a category drawn around the
#: pages that happen to score badly would guarantee its own result. The rules
#: that needed a decision:
#:
#:   * A publisher is `news` when the landing page's substance is editorial,
#:     including portals where that is true (163.com, aol.com, mail.ru,
#:     dzen.ru, naver.com, onet.pl, qq.com, seznam.cz, sina.com.cn, sohu.com,
#:     t-online.de, uol.com.br, wp.pl, yahoo.com, yandex.ru).
#:   * A platform is `social` even when it is ad-supported and content-heavy
#:     (substack.com, imgur.com, scribd.com, wattpad.com, twitter.com),
#:     because the page being priced is a feed or a viewer, not an article.
#:   * `infrastructure` is for domains with no page to speak of: CDN roots,
#:     DNS and NTP endpoints, ad-exchange hostnames, URL shorteners, parked
#:     and redirect-only domains. They are in the Tranco top 500 on request
#:     volume, not on traffic to a document, and they behave unlike anything
#:     else here.
#:   * `productivity` is the widest: SaaS, developer tools, cloud, corporate
#:     sites, telecoms, banks and other business services.
#:
#: Every crawled domain appears exactly once, which
#: `test_every_crawled_domain_has_a_category` enforces.
CATEGORY_FILE = ROOT / "data" / "tranco_500_categories.csv"

CATEGORIES: tuple[str, ...] = (
    "entertainment", "infrastructure", "news", "productivity", "reference",
    "shopping", "social",
)


@functools.cache
def load_categories() -> dict[str, str]:
    """Registrable domain -> category, for the crawled 500."""
    with open(CATEGORY_FILE, newline="") as f:
        out = {r["domain"]: r["category"] for r in csv.DictReader(f)}
    unknown = sorted(set(out.values()) - set(CATEGORIES))
    assert not unknown, f"{CATEGORY_FILE} uses categories not in CATEGORIES: {unknown}"
    return out


def category_for(page_url: str) -> str:
    """Which category a crawled page belongs to.

    Falls back to `infrastructure` for a host the file does not name, which
    cannot happen for the crawled 500 and keeps this total for anything else a
    caller passes in.
    """
    host = urlparse(page_url).netloc or page_url
    host = host.split("@")[-1].split(":")[0].lower()
    cats = load_categories()
    return cats.get(host) or cats.get(host.removeprefix("www."), "infrastructure")


# A category that misses `Bounds.max_abs_bytes_bias_pct` must be recorded here
# with its measured bias, exactly as `solutions.KNOWN_UNCALIBRATED` records a
# miss on the journey suite, and only the categories in
# `BIASED_CATEGORIES_SKIPPED` are let off with a skip. Widening the bound to
# fit a category would defeat the point of having them.
KNOWN_BIASED_CATEGORIES: dict[str, str] = {
    "infrastructure": (
        "+33.7% over 1,171 matched requests carrying 11.7 MB, which is 6% of "
        "the crawl's priced bytes. Three pages account for most of it "
        "(imgsmail.ru, cdnvideo.ru, fwmrm.net) and three hosts do within them "
        "(yandex.ru, snap.licdn.com, mc.yandex.ru): on a CDN root or an "
        "ad-exchange hostname the few tracker scripts that load come back "
        "far smaller than the same URL does on a real page, and the table "
        "answers with the population mean. Recorded rather than fixed: it is "
        "the one category whose pages are not documents, and pulling the "
        "table toward them would cost accuracy everywhere that matters."
    ),
}

#: Categories whose miss is recorded rather than failing the suite.
#:
#: `news` was here at -18.7%, and the fix was the host rungs in
#: `llm-classifier`: the category's gap was almost entirely ad and
#: content-recommendation *scripts*, whose vendors serve versioned bundles from
#: per-customer subdomains, so every one of them missed the path rungs and was
#: answered with the mean of any .js (37 KB) instead of its own host's (up to
#: 179 KB). It now measures -1.5%, inside the bound. See `Level::Host` in
#: llm-classifier/src/lib.rs.
BIASED_CATEGORIES_SKIPPED: tuple[str, ...] = ("infrastructure",)


# --------------------------------------------------------------------------- #
# Tracker roles
# --------------------------------------------------------------------------- #
#: What kind of tracker a blocked request is, as `family-kind`.
#:
#: The page categories above cut the crawl by the *page* a request was blocked
#: on. This cuts it by the request, and it exists because the two axes hide
#: different errors. A page cut asks whether the estimator is as good on a
#: bank as on a newspaper; this asks whether it is as good on an ad loader as
#: on an analytics beacon. Neither implies the other, and on this crawl the
#: second is where the error is: the aggregate reads +2.1% and is the sum of
#: six roles contributing between -2.0 and +2.7 points, which is cancellation
#: rather than accuracy.
#:
#: BOTH HALVES ARE MEASURED, which is the property that makes this worth
#: having. `family` is the Disconnect list's own category for the URL, via
#: `disconnect.tracker_category` -- the same list Firefox consults, decided by
#: someone else, for reasons that have nothing to do with this repo. `kind` is
#: the request context the estimator already keys on, so it is whatever
#: Playwright said the request was for. Nothing here was drawn around the
#: requests that happen to score badly, which is the trap the page-category
#: comment above names and the one a taxonomy invented after looking at the
#: errors would fall straight into.
#:
#: A functional taxonomy was the first thing tried and it is not shippable on
#: the signals available, which is worth recording so it is not tried again
#: blind. The distinction that matters for the cascade is loader against
#: leaf -- does this script pull a subtree in behind it -- and the obvious
#: measured proxy is rootness, the share of a host's requests the markup asks
#: for directly, from HTTP Archive's `initiator_type`. It does not survive
#: contact with the data: `connect.facebook.net`, the canonical loader, scores
#: 0.03, while `static.cloudflareinsights.com`, a beacon, scores 0.99. Modern
#: tag setups inject everything from script, so rootness measures how a vendor
#: is installed rather than what it does. The remaining option was hand-written
#: URL rules, which would have been drawn from the errors below.
#:
#: `family` collapses the list's categories onto shorter names, and leaves
#: anything else -- including a URL the list does not name, which happens when
#: Firefox blocked on a table the shipped list has since dropped -- as
#: `other`. `kind` collapses the contexts the same way: `script`, `frame`
#: (an ad or social iframe), `pixel` (an image) and `beacon` (everything
#: else, which on this crawl is XHR, fetch and ping).
TRACKER_FAMILIES: dict[str, str] = {
    "Advertising": "ad",
    "Analytics": "analytics",
    "Social": "social",
    "Content": "content",
    "FingerprintingInvasive": "fingerprinting",
}

TRACKER_KINDS: dict[str, str] = {
    "SCRIPT": "script",
    "HTML": "frame",
    "IMAGE": "pixel",
    "OTHER": "beacon",
}


@functools.cache
def tracker_role_for(url: str, resource_type: str) -> str:
    """`family-kind` for one blocked request.

    Cached because the pooled crawl holds ten passes over the same pages, so
    every URL arrives about ten times and the list lookup is the only cost
    this adds.

    Falls back to `other` on either half rather than raising, so a list
    update that introduces a category, or a crawl that records a context this
    module has not seen, shows up as mass moving into `other-*` instead of as
    an error. `test_tracker_roles_cover_the_crawl` is what notices.
    """
    from compare_estimate_vs_etp import context_for

    try:
        import disconnect
        family = disconnect.tracker_category(url)
    except Exception:
        family = None
    context = str(context_for(resource_type)).rsplit(".", 1)[-1]
    return (f"{TRACKER_FAMILIES.get(str(family), 'other')}-"
            f"{TRACKER_KINDS.get(context, 'other')}")


def load_blocked_by_tracker_role() -> dict[str, list[BlockedRequest]]:
    """The priced requests, grouped by what kind of tracker they are."""
    out: dict[str, list[BlockedRequest]] = {}
    for r in load_blocked_requests():
        out.setdefault(tracker_role_for(r.url, r.resource_type), []).append(r)
    return out


#: Roles whose byte bias misses `Bounds.max_abs_bytes_bias_pct`, with the
#: measured value recorded beside them, exactly as `KNOWN_BIASED_CATEGORIES`
#: records a page category that misses.
#:
#: This registry is load-bearing in a way that one is not: the value is
#: asserted, to `Bounds.known_tracker_role_bias_tolerance_pct`, so a recorded
#: miss that drifts fails rather than staying quietly recorded. A bias nobody
#: intends to fix should still not be free to move.
#:
#: Both entries are the same kind of thing and it is not a table bug. The
#: shipped size table is fitted on HTTP Archive; these are populations where
#: HTTP Archive and this crawl disagree about the same URL.
#:
#:   * `social-script` is almost entirely `snap.licdn.com`. The table answers
#:     20.3 kB for `/li.lms-analytics/insight.min.js`, and HTTP Archive backs
#:     it: 20.2 kB over 7,427 observations of that exact path. This crawl
#:     measures 1.9 kB, 322 times, without varying. Two crawls, one URL, a
#:     factor of ten. Nothing in the estimator is wrong here and pulling the
#:     table to 1.9 kB would be fitting it to the test.
#:   * `ad-beacon` is the opposite end of the same problem. A beacon's body
#:     is a few hundred bytes of payload whose size depends on what the page
#:     put in it, so the conditional mean the table returns is right on
#:     average and wrong on almost every instance. It carries 4.1% of the
#:     crawl's priced bytes, so +41.6% of it is +1.7 points on the total.
#:
#: `fingerprinting-script` is a third, larger miss at -83.9%, and is not here
#: because it is not gradeable: 99 priced requests, all of them
#: `tags.tiqcdn.com`. See `Bounds.gradeable_tracker_role_requests`.
KNOWN_BIASED_TRACKER_ROLES: dict[str, float] = {
    "social-script": 16.8,
    "ad-beacon": 41.6,
}

# The tracker-role twin of `COMPOSITION_SKEW`, and the reason the axis earns
# its place rather than duplicating the page cut. It lifts `ad-script` by 15%
# and drops `analytics-script` by the same number of bytes, so:
#
#   * the aggregate does not move at all, by construction;
#   * no page category moves outside `max_abs_bytes_bias_pct` -- the worst
#     goes to 7.6%, because both roles are spread over every kind of page;
#   * the moved role contributes 6.4% of the crawl's observed bytes in error,
#     against a cap of 4.0%.
#
# So it is invisible to every bound that existed before this axis and caught
# by the one that came with it. 15% is the smallest round skew that does
# that: at 12% the page cut still passes but the contribution is 4.8%, and at
# 20% `reference` and `social` start to fail the page cut, which would make
# the control prove less than it claims.
TRACKER_COMPOSITION_SKEW: tuple[str, str, float] = (
    "ad-script", "analytics-script", 0.15)


# --------------------------------------------------------------------------- #
# Match levels
# --------------------------------------------------------------------------- #
#: The rungs of the estimator's fallback hierarchy, most specific first.
#:
#: A third axis, and the sharpest of the three. `CATEGORIES` cuts by the page
#: a request was blocked on and `TRACKER_FAMILIES` by what kind of tracker it
#: was; this cuts by *how specifically the estimator recognised the URL* --
#: whether it answered from the exact path it had seen before, from the
#: vendor's host, or from the mean of every `.js` in HTTP Archive.
#: `llm_classifier.classify_url` reports it, out of the same table walk that
#: produces the estimate.
#:
#: This is the axis with the least room to argue about its independence. The
#: page categories are hand-labelled, and the tracker roles lean on the
#: Disconnect list's judgement; the rung a URL lands on is decided by
#: `llm-classifier/scripts/build_table.py` out of HTTP Archive, before any
#: crawl exists to grade against, and nothing about this crawl can move it.
#:
#: It is also the axis that finds the most. The rungs are not equally
#: calibrated and the error is monotone in specificity: the table
#: over-answers where it recognises the URL and under-answers where it does
#: not, from +55.8% at `template` through +13.8% at `prefix` to -24.3% at
#: `ext_query` and -98.1% at `context`. Weighted by mass that is +5.2 points
#: of the crawl's bytes at `prefix` against -2.6 at `ext_query`, inside an
#: aggregate that reads +2.1%.
#:
#: Some of that is selection rather than defect, and the distinction matters
#: for how to read a failure. A URL lands on `prefix` *because* it had no
#: `path` entry, so conditioning on the rung conditions on the table's
#: coverage, which correlates with how common and how large the asset is.
#: What makes it worth bounding anyway is that the aggregate is then right
#: only by cancellation, and the mix of rungs in a real browsing session is
#: not the mix in this crawl -- so a total that depends on the mix is a total
#: that can move for reasons nothing here would see.
#: `FOLLOWUP_BYTES_PER_REQUEST` for SCRIPT and HTML, mirrored from
#: `llm-classifier/src/lib.rs` so `test_the_cascade_modulation_is_levelled`
#: can price the cascade against the flat constant the per-rung factors are
#: supposed to average to. A copy, with the same discipline as
#: `fit_followups.SHIPPED_BYTES_PER_REQUEST`: if one moves, this must.
FOLLOWUP_BYTES_PER_REQUEST: int = 47_000

MATCH_LEVELS: tuple[str, ...] = (
    "path", "template", "prefix", "host", "host_template", "ext_query",
    "ext", "context", "bodyless",
)


@functools.cache
def match_level_for(url: str, resource_type: str) -> str:
    """Which rung of the estimator's table answered for this request.

    Cached for the reason `tracker_role_for` is: ten passes over the same
    pages means every URL arrives about ten times.

    Passes an empty method, as `predict` does, because the crawl did not
    record one -- so the `bodyless` rung is never reached here and the
    estimator reads every request as a GET. See `predict`.
    """
    import llm_classifier
    from compare_estimate_vs_etp import context_for

    return llm_classifier.classify_url(
        url, context_for(resource_type),
        llm_classifier.RequestInitiator.UNKNOWN, "")


def load_blocked_by_match_level() -> dict[str, list[BlockedRequest]]:
    """The priced requests, grouped by the rung that answered for them."""
    out: dict[str, list[BlockedRequest]] = {}
    for r in load_blocked_requests():
        out.setdefault(match_level_for(r.url, r.resource_type), []).append(r)
    return out


#: The measured byte bias of every gradeable rung, asserted rather than
#: merely recorded.
#:
#: This goes further than `KNOWN_BIASED_TRACKER_ROLES`, which records only
#: the roles that miss the bound, and the difference is deliberate. A tracker
#: role's bias is partly a fact about the world -- how Tealium deploys, how
#: LinkedIn serves its insight tag -- so pinning a role that currently passes
#: would be pinning the population. A rung's bias is a fact about
#: `build_table.py`: which URLs got their own entry and which fell through.
#: Every rung is therefore a calibration target, and every gradeable one is
#: pinned here whether it passes the bound or not.
#:
#: That closes a gap the contribution cap cannot. `prefix` already spends
#: 5.2 of the 7.0 points that cap allows, so it has no room left to catch a
#: drift on another rung: `host_template` could go from -4.9% to +9.4% --
#: a 15% shift in the estimator, 4.2 points of the crawl's bytes -- and pass
#: both the cap and the 12% bar, because it passes the bar today. Pinned, it
#: fails.
#:
#: Three of the four miss `Bounds.max_abs_bytes_bias_pct`, which is the
#: finding rather than a defect of the bound: the fallback chain is
#: calibrated as a whole and not rung by rung. These are also the only
#: recorded biases in this module a table rebuild could actually close --
#: the page-category and tracker-role ones are two crawls disagreeing about
#: the same URL, and no table change fixes that.
MATCH_LEVEL_BIAS: dict[str, float] = {
    "template": 55.8,
    "prefix": 13.8,
    "host_template": -4.9,
    "ext_query": -24.3,
}


# --------------------------------------------------------------------------- #
# An outside opinion
# --------------------------------------------------------------------------- #
#: What HTTP Archive's own crawl saw on the same domains.
#:
#: Every other number in this module comes from one paired crawl run on one
#: afternoon, and every bound divides by something that crawl measured. This
#: is the one denominator from somewhere else: HTTP Archive crawled the same
#: sites with a different browser on a different day, and
#: `src/build_tranco_500_http_archive.py` reduces its 50%-sampled request
#: export to one row per domain. 349 of the 500 are in it.
#:
#: What it is good for and what it is not. It measures a page's tracker mass
#: directly -- a sum over requests -- where the crawl measures a difference
#: between two whole page loads, so it does not carry the churn that makes
#: `bytes_saved` so noisy. But it is a different browser on a different page
#: state: its pages are heavier, it counts trackers ETP would not block
#: (an entity-list exception is still a Disconnect entry), and it misses what
#: a blocked tracker would have pulled in from a host the list does not name.
#: So the ratio against it is nowhere near 1 and is not supposed to be; what
#: it certifies is that the estimator's total is the right size against a
#: population it has never seen, which nothing else here can say.
HTTP_ARCHIVE_CSV = ROOT / "data" / "tranco_500_http_archive.csv"


@dataclass(frozen=True)
class HttpArchivePage:
    """One domain, as HTTP Archive's crawl of it came out.

    Counts and bytes are of the *sampled* half of the requests, which is why
    nothing reads them as absolute totals; the tracker share and the ratio a
    bound divides by are unaffected by the sampling.
    """

    domain: str
    n_req: int
    bytes: int
    tracker_bytes: int
    n_tracker_req: int
    n_tracker_hosts: int


@functools.cache
def load_http_archive_mass() -> dict[str, HttpArchivePage]:
    """Registrable domain -> HTTP Archive's measurement of it, or empty."""
    if not HTTP_ARCHIVE_CSV.exists():
        return {}
    with open(HTTP_ARCHIVE_CSV, newline="") as f:
        return {r["domain"]: HttpArchivePage(
            domain=r["domain"],
            n_req=int(r["n_req"]),
            bytes=int(r["bytes"]),
            tracker_bytes=int(r["tracker_bytes"]),
            n_tracker_req=int(r["n_tracker_req"]),
            n_tracker_hosts=int(r["n_tracker_hosts"]),
        ) for r in csv.DictReader(f)}


def http_archive_for(page: CrawledPage) -> HttpArchivePage | None:
    """HTTP Archive's row for a crawled page, if it has one."""
    host = urlparse(page.url).netloc.split("@")[-1].split(":")[0].lower()
    mass = load_http_archive_mass()
    return mass.get(host) or mass.get(host.removeprefix("www."))


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def predict(requests: list[BlockedRequest],
            include_followups: bool = False) -> tuple[list[int], list[float]]:
    """(bytes, cpu_ms) from the shipped table, per request.

    The estimator sees only the URL and the request context -- what Firefox has
    before a response exists -- so nothing here leaks the observed size.

    `include_followups` picks which of the two costs the estimator offers, and
    the choice is dictated by the ground truth each test has. A control-arm
    `_transferSize` is one response, so the per-request and per-page byte
    checks compare against the direct estimate and pass it false. The
    page-level delta between the two arms is everything the page did not fetch,
    the pruned subtree included, so the tests that divide by it pass it true.
    Comparing a direct estimate against a page delta is what made
    `Bounds.bytes_page_ratio_min` a statement about the size of the cascade
    rather than about the estimator, which is what it no longer is.

    The CPU tests are the exception, and deliberately: they are scored against
    a page-level measurement but still use the direct estimate. See
    `Bounds.cpu_ratio_max`.

    It also takes an initiator and a method, and this crawl recorded neither:
    `firefox_crawl_500_tracking.py` logs the blocked URL and its resource type
    and nothing else about the request. So both are passed as "not known",
    which is the estimator's pre-existing behaviour -- the table has no use for
    the initiator in any case, and an unknown method reads it as a GET. The
    consequence for these tests is that the bodyless-method rule is not
    exercised here; the journey suite is where it is measured.
    """
    import llm_classifier
    from compare_estimate_vs_etp import context_for

    sizes: list[int] = []
    cpu: list[float] = []
    for r in requests:
        b, ms = llm_classifier.estimate_resources(
            r.url, context_for(r.resource_type),
            llm_classifier.RequestInitiator.UNKNOWN, "", include_followups)
        sizes.append(b)
        cpu.append(ms)
    return sizes, cpu


@functools.cache
def matched_totals_by_page() -> dict[int, tuple[int, int]]:
    """(predicted, observed) direct bytes per page, over the priced blocks.

    Cached because the subset tests ask for hundreds of overlapping subtotals
    and the prediction for a request does not depend on which subset it is
    being counted in.
    """
    out: dict[int, list[int]] = {}
    requests = load_blocked_requests()
    sizes, _cpu = predict(requests)
    for r, size in zip(requests, sizes):
        acc = out.setdefault(r.page_idx, [0, 0])
        acc[0] += size
        acc[1] += r.observed_bytes
    return {k: (v[0], v[1]) for k, v in out.items()}


@functools.cache
def followup_totals_by_page() -> dict[int, int]:
    """Cascade-inclusive predicted bytes per page, over *all* tracking blocks."""
    out: dict[int, int] = {}
    requests = load_all_tracking_blocks()
    sizes, _cpu = predict(requests, include_followups=True)
    for r, size in zip(requests, sizes):
        out[r.page_idx] = out.get(r.page_idx, 0) + size
    return out


# Deliberately wrong predictors, as a negative control on the bounds below. If
# a bug made the comparison insensitive to prediction quality, these would pass
# it, so the tests assert that they do not.
BROKEN_SCALES: dict[str, float] = {
    "zeroed": 0.0,
    "tenfold": 10.0,
    "tenth": 0.1,
}

# The negative control the scales above cannot be: every one of them moves the
# whole-crawl total, so the aggregate bounds catch them and a bound that only
# worked per category would look redundant. This one is invisible to a total
# by construction -- it lifts the estimate on `news` and drops it on
# `productivity` by the same number of bytes, which is the error a single
# ratio over 271 pages is blind to and the reason
# `Bounds.category_tracking_ratio_tolerance_pct` exists. 30% is enough: it
# leaves the whole-crawl ratio at 1.04 -- it moves the same bytes off one
# category and onto the other -- and takes news to 1.24 (+30%) and
# productivity to 0.78 (-24%), both outside tolerance.
COMPOSITION_SKEW: tuple[str, str, float] = ("news", "productivity", 0.30)

#: Ratio of the cascade-inclusive estimate to the raw Disconnect-only delta,
#: per category, as measured -- the centres
#: `Bounds.category_tracking_ratio_tolerance_pct` is a tolerance around.
#:
#: Only the categories whose delta is quiet enough to grade are here, and
#: which those are is itself measured: `tracking_churn_se_pct` against
#: `Bounds.gradeable_tracking_se_pct`. A category that becomes gradeable, or
#: stops being, fails `test_tracking_ratio_holds_per_category` rather than
#: being silently skipped -- the set is part of what is being asserted.
#:
#: These are records, not targets. They sit either side of the crawl-wide
#: 1.04 and much closer to it than the single-pass crawl's 1.26 and 1.61 sat
#: to its 1.47, which is what recalibrating the cascade against this same
#: denominator does and is not evidence that the categories agree. Nothing
#: here says which is right: the denominator sees only the part of the
#: cascade that lands back on listed hosts, and there is no reason that share
#: is the same on a news site as in a web console. What they are for is
#: noticing when the estimator moves one of them and not the other.
CATEGORY_TRACKING_RATIO: dict[str, float] = {
    "news": 0.95,
    "productivity": 1.03,
}


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bounds:
    """How far off the top-500 estimates are allowed to be.

    Set off the measured values with headroom, so a failure means the estimator
    moved rather than that the sample did -- and the sample cannot move here,
    since both tables are fixed files.
    """

    #: Cap on |sum(predicted) / sum(observed) - 1| over blocked requests.
    #: Measured +2.1% over the 11,290 priced blocks in the pooled crawl; the
    #: cap matches the journey suite's MAX_SIGNED_BIAS_PCT.
    #:
    #: It is the same 12% it was on the single-pass crawl, where the figure
    #: read -2.5%, and the reason repeating the crawl did not buy a tighter
    #: bound here is worth stating once because it applies to half of what
    #: follows. This is a per-*request* comparison: a blocked URL against the
    #: size that same URL transferred in the other arm. Ten passes give ten
    #: times the observations, but they are ten observations of the same
    #: 1,875 requests on the same 271 pages, and what limits the bound is
    #: which requests were crawled, not how precisely each was weighed.
    #: Repetition helps the page-level denominators, which are differences of
    #: two noisy totals; it barely moves this one.
    max_abs_bytes_bias_pct: float = 12.0

    #: Floor on the share of pages whose predicted byte total lands within 25%
    #: of the observed total. Measured 0.435 over 255 pages.
    min_pages_within_25pct: float = 0.35

    # ----------------------------------------------------------------- #
    # Tracker roles
    # ----------------------------------------------------------------- #
    # The bounds above and below cut the crawl by page. These cut it by what
    # kind of tracker was blocked -- see `TRACKER_FAMILIES` for the taxonomy
    # and why both halves of it are measured rather than invented.

    #: Cap on how much of the crawl's observed bytes any one tracker role may
    #: contribute in error: |predicted - observed| over that role, as a share
    #: of the *whole* crawl's observed total. Measured worst 2.7%
    #: (`social-script`).
    #:
    #: This is the primary bound on the axis and it is deliberately not a
    #: relative bias. A relative bar asks every role to be equally accurate,
    #: which is the wrong question twice over: it is unmeetable on a role
    #: whose requests are 1 kB beacons, and it is far too lax on a role
    #: carrying half the crawl. What the estimator promises is an unbiased
    #: *total*, so the thing to bound is how much each role can push that
    #: total around -- which scales the bar with the role's mass
    #: automatically, and needs no exemption list.
    #:
    #: The six roles above 0.5% of the crawl contribute -1.9, +1.4, +2.7,
    #: +1.7, -2.0 and +0.8 points, summing to the +2.1% the aggregate
    #: reports. That is the case for the axis in one line: the aggregate is
    #: not accurate, it is cancelling, and nothing that existed before this
    #: bound could see the difference.
    #:
    #: 4.0% is set off the 2.7% measured with enough headroom that a role
    #: drifting slowly does not fail on noise, and it catches a composition
    #: error of about 12% between the two largest script roles.
    #: `TRACKER_COMPOSITION_SKEW` records where that line is, at 15%, by
    #: failing when this bound stops catching it.
    max_tracker_role_bias_contribution_pct: float = 4.0

    #: Floors a role must clear before its *relative* bias is graded against
    #: `max_abs_bytes_bias_pct`: a share of the crawl's priced bytes, and a
    #: count of priced requests.
    #:
    #: Both are needed and they exclude different things. The byte floor
    #: drops roles too small to matter for the total, like `analytics-pixel`
    #: at 0.04%. The request floor drops roles that are large enough to
    #: matter and are nonetheless one vendor: `fingerprinting-script` carries
    #: 2.4% of the crawl's priced bytes and is 99 requests, every one of them
    #: `tags.tiqcdn.com`. Its -83.9% is a statement about how Tealium's tag
    #: payloads differ between this crawl and HTTP Archive, not about the
    #: estimator, and grading it would make the suite's verdict depend on one
    #: vendor's deployment. It still shows up in the contribution bound above
    #: if it ever grows past 4%.
    #:
    #: Four roles clear both: `ad-script` (-3.6%), `analytics-script`
    #: (+6.9%), `social-script` (+16.8%) and `ad-beacon` (+41.6%). The last
    #: two are recorded in `KNOWN_BIASED_TRACKER_ROLES`.
    gradeable_tracker_role_byte_share: float = 0.02
    gradeable_tracker_role_requests: int = 200

    #: The same pair for the match-level axis, and the same reasoning. Four
    #: rungs clear both: `template` (+55.8%), `prefix` (+13.8%),
    #: `host_template` (-4.9%) and `ext_query` (-24.3%). `ext` and `context`
    #: miss by more than any of them -- -87.6% and -98.1% -- and are excluded
    #: by mass, which is the right call twice over: together they are 1.2% of
    #: the crawl's priced bytes, and a rung that answers with the mean of
    #: every `.js` in HTTP Archive is *supposed* to be badly wrong per
    #: request. They stay inside the contribution bound, which is where they
    #: would show up if they grew.
    gradeable_match_level_byte_share: float = 0.02
    gradeable_match_level_requests: int = 200

    #: Cap on how much of the crawl's observed bytes any one rung of the
    #: estimator's fallback hierarchy may contribute in error. Measured worst
    #: +5.2% (`prefix`), against +2.7% for the worst tracker role and +2.1%
    #: for the whole crawl.
    #:
    #: The sharpest of the three composition bounds, and the one whose
    #: failure would be most actionable: unlike a page category or a tracker
    #: role, a rung is something `build_table.py` controls directly.
    #:
    #: 7.0 sits off the 5.2 measured with room for an ordinary table rebuild
    #: to move the mix. It is looser in absolute terms than the tracker-role
    #: cap of 4.0 and tighter relative to what it is bounding, which is the
    #: honest shape of the measurement rather than a preference.
    max_match_level_bias_contribution_pct: float = 7.0


    #: How far a bias recorded in `KNOWN_BIASED_TRACKER_ROLES` may drift from
    #: its recorded value, in percentage points of that bias.
    #:
    #: The page-category registry records a miss and then stops caring what
    #: it does; this one does not, because a recorded number that nothing
    #: checks decays into folklore. 25% of the recorded value is wide enough
    #: that the two entries have room to move with an ordinary table rebuild
    #: and narrow enough that `social-script` could not drift to `ad-beacon`'s
    #: +41.6% without saying so.
    known_tracker_role_bias_tolerance_pct: float = 25.0

    #: Predicted CPU over *all* tracking blocks, as a fraction of the CPU
    #: measured at page level. Measured 0.55, from the direct estimate.
    #:
    #: The byte bands above switched to the cascade-inclusive estimate when
    #: `estimate_resources` grew `include_followups`; this one did not, and the
    #: reason is that it would have nothing left to measure. The follow-up CPU
    #: rate was calibrated to the page-level CPU saving this band divides by,
    #: so a cascade-inclusive CPU total reads 0.92 here by construction --
    #: a test that would be passing because a constant was set to make it
    #: pass. At 0.55 the direct estimate is at least being compared against
    #: something it was not fitted to.
    #:
    #: Read this band more sceptically than the byte ones, because repeating
    #: the crawl made the CPU instrument *worse* rather than better and the
    #: numbers say by how much. Pooling the passes measures the standard error
    #: of the CPU saving directly, where a single pass could only infer it:
    #: it is 54%. Worse, the raw delta over the pages ETP touched is -890 s
    #: across ten passes -- the blocking arm burned more CPU than the control
    #: arm -- and what makes `cpu_saved_s` positive at all is the drift
    #: correction, which is -6.2 s per untouched page and so worth +1,687 s
    #: over the 271 touched ones. A correction twice the size of the result it
    #: produces is not a measurement, and the cause is structural: this crawl
    #: runs its whole control arm before its whole blocking arm, so a page's
    #: two loads are 133 minutes and a lot of machine state apart. Bytes
    #: barely notice; CPU does. An interleaved crawl would fix it.
    #:
    #: The ceiling is 1.0 on principle rather than from the measurement: a
    #: per-request estimate must not exceed the page-level saving, because
    #: blocking a tracker also prevents the subresources it would have
    #: requested and executed, and the estimator does not model those.
    #: Predicting more CPU than was measurably saved would mean it is
    #: over-attributing. Note that the byte twin of that argument did not
    #: survive this crawl -- see `bytes_page_ratio_max_raw` -- and the only
    #: reason it survives here is that no quiet CPU instrument exists to
    #: contradict it, which is not the same as evidence for it.
    #:
    #: The floor guards the other way -- against the coefficients collapsing to
    #: nothing, which would otherwise look like a pass.
    #:
    #: Be clear about the sensitivity this buys: with the measurement at 0.55,
    #: the band tolerates scaling every CPU estimate by anything from 0.45x to
    #: 1.8x, and a 54% standard error on the denominator means even that is
    #: generous. It catches the model breaking, not the coefficients being
    #: somewhat wrong -- and it cannot do better while `cpu_ms` is derived
    #: rather than fitted and the only ground truth is per page. The negative
    #: control in `test_broken_estimates_are_detectably_bad` records where the
    #: line is.
    cpu_ratio_min: float = 0.25
    cpu_ratio_max: float = 1.0

    #: Predicted bytes over *all* tracking blocks, as a fraction of the bytes
    #: measured saved over the whole page, with `include_followups`. Measured
    #: 1.34 against the drift-corrected 435.0 MB and 1.48 against the raw
    #: 392.8 MB, both ten passes' worth.
    #:
    #: This is the byte twin of the CPU band, and exists to say something
    #: about the 7,459 blocks that `load_blocked_requests` cannot price. It is
    #: also, as of the ten-pass crawl, the module's *weakest* page-level
    #: statement rather than one of its two strongest, and the demotion is the
    #: single biggest thing repeating the crawl changed. The rest of this
    #: comment is why, because a reader who takes these bands at their old
    #: meaning will draw the wrong conclusion from a failure.
    #:
    #: WHAT THE CEILING USED TO MEAN. A ceiling of 1.0 was structural. The
    #: measurement is everything the blocking arm did not fetch; the estimate
    #: is what the estimator thinks blocking removed; so a ratio above 1.0
    #: could only mean over-attribution, whatever the crawl's noise. On the
    #: single-pass crawl it measured 0.84 and the argument was never tested.
    #:
    #: WHY IT IS NOT STRUCTURAL. Because the premise is false on this crawl,
    #: and the pooled tables show it two independent ways.
    #:
    #:   * The whole-page delta is 392.8 MB. The Disconnect-only delta --
    #:     literally a subset of the same difference, counted over fewer
    #:     requests -- is 559.4 MB. A subset cannot exceed its superset, so
    #:     the blocking arm must be fetching *more* of something the list does
    #:     not name: 166.6 MB more across ten passes, at t = -2.1.
    #:   * One page shows the mechanism at full size. tradingview.com's
    #:     blocking arm transferred 106.0 MB more than its control arm over
    #:     ten passes, which is 27% of the whole-page denominator on its own.
    #:     It is a streaming charting app; what differs between two loads of
    #:     it two hours apart has nothing to do with ETP.
    #:
    #: Dropping that page is not a rescue -- the ratio only falls from 1.48 to
    #: 1.17, still over the old ceiling -- so this is not one outlier. It is
    #: that a whole-page difference is not a measurement of blocking on pages
    #: whose first-party content moves by more than blocking does.
    #:
    #: WHAT THE BANDS ARE NOW. [1.05, 1.91] around the measured 1.48: one
    #: standard error of the denominator either way, jackknifed over the ten
    #: passes at 29.1%. They were [0.85, 2.0], about 1.5 of those, and were
    #: described here as factor-level sanity checks; at one they are no
    #: longer that, and they now catch a 29% scaling error where they needed
    #: 43%.
    #:
    #: The trade is explicit and it is not free. A standard error is a
    #: standard error: repeat the crawl and this ratio lands outside the band
    #: about a third of the time with nothing wrong. That is tolerable here
    #: only because the crawl is a fixed artefact checked into the analysis
    #: rather than something the suite re-measures, so the band is really a
    #: guard against the *estimator* moving; if the crawl is ever replaced,
    #: these three bands have to be re-derived from the new passes before any
    #: failure they report means anything. Per pass this ratio runs from
    #: -4.1 to +8.7, which is the scale of what the pooling is doing.
    #:
    #: They are kept, rather than deleted as redundant, for one reason: this
    #: is the only page-level denominator that contains the unlisted bytes at
    #: all. The quiet instrument below is quiet precisely because it throws
    #: them away, so if the estimator ever grew a term for the unlisted half
    #: of the cascade, this pair is the only thing here that would react.
    #:
    #: AND THE DEMOTION HAS SINCE BEEN UNDONE, though not for this pair. Both
    #: bullets above are about the same six pages, and `page_delta_instability`
    #: identifies them from the spread of their per-pass deltas without
    #: reference to the estimator: without them the whole-page delta is
    #: 542.2 MB, the subset relation holds again, and the ratio is 1.07.
    #: `bytes_page_ratio_max_stable` is that bound and is the one to read.
    #: This pair stays unfiltered and wide because it is the version that
    #: assumes nothing about which pages the crawl can measure, and a
    #: filtered bound should not be the only page-level statement in the
    #: module that covers every page.
    bytes_page_ratio_min: float = 1.05
    bytes_page_ratio_max: float = 1.85
    bytes_page_ratio_max_raw: float = 1.91

    #: The same prediction over `MeasuredPageCost.bytes_saved_raw_stable`:
    #: the raw whole-page delta over the 265 of 271 touched pages whose loads
    #: repeat. Measured 1.07.
    #:
    #: This is what the pair above becomes once the crawl's own per-pass
    #: tables are used to say which pages the whole-page delta can measure at
    #: all, and it is the improvement the ten passes were run for and did not
    #: at first deliver. The comments above, and the module docstring, say the
    #: repetition bought nothing here. That was true of what was done with it:
    #: the passes were pooled into one total and then used only to put a
    #: standard error on it. Pooling cannot remove a page whose two loads
    #: differ by 106 MB for reasons that have nothing to do with blocking;
    #: it averages it in ten times. Reading the passes per page does remove
    #: it, and there is no other way to identify it, which makes this the one
    #: page-level bound in the module that a single-pass crawl could not have
    #: stated.
    #:
    #: WHAT IT BUYS, against the unfiltered pair, in the three things a bound
    #: is worth judging on:
    #:
    #:   * Centre. 1.07 against 1.48. The estimator has not moved; the
    #:     denominator has, from 392.8 MB to 542.2 MB.
    #:   * Precision. The denominator's standard error over the passes falls
    #:     from 23.4% to 16.2%, and the ratio's own jackknife over passes
    #:     from 23.6% to 18.6%.
    #:   * Agreement. It lands on the quiet instrument's answer -- 1.07 here
    #:     against 1.05 for `bytes_tracking_ratio_*` on the same pages --
    #:     where the unfiltered pair is 41% above it. Two denominators that
    #:     share a numerator and disagree by 41% are not both measuring the
    #:     estimator.
    #:
    #: WHAT IT COSTS. 0.7% of the predicted bytes and six pages, and the
    #: filter is a page-selection step that has to be trusted. It is built
    #: not to need much trust: `page_delta_instability` scores a page on the
    #: spread of its deltas over its own size and never on their level, so no
    #: page can be dropped for disagreeing with the estimator, and the
    #: threshold was fixed at 1.0 out of a range it is flat over -- 0.5 to
    #: 2.0 puts this ratio between 1.05 and 1.15, on page sets of 259 to 269.
    #: `bytes_page_ratio_max_raw` stays beside it, unfiltered and wide, so the
    #: module still states the version of this that assumes nothing.
    #:
    #: THE BAND IS [0.86, 1.27], one standard error of the denominator either
    #: way. It was [0.80, 1.35], about 1.5, and the narrowing is a deliberate
    #: trade rather than a better measurement: the error bar did not move.
    #: The denominator's standard error is measured here by jackknifing the
    #: pooled ratio over the ten passes -- leave one out, re-pool, repeat --
    #: which gives 19.3% against the 16.2% the per-pass spread implies, so
    #: this band is the more conservative of the two readings.
    #:
    #: WHAT ONE STANDARD ERROR COSTS. Re-crawling would move this ratio by
    #: more than a standard error about a third of the time, so a failure
    #: here is now weaker evidence of a regression than it was and the first
    #: thing to check is whether the crawl was replaced. What it buys is that
    #: scaling every estimate by anything outside [0.81, 1.19] is caught,
    #: against [0.75, 1.26] before. The band no longer covers the instability
    #: threshold's own range: 0.5 to 2.0 puts this ratio between 1.05 and
    #: 1.15, which still fits, but with less room than the old band had.
    #:
    #: It also keeps one thing the sharp bound gave up. `FOLLOWUP_BYTES_PER_REQUEST`
    #: was calibrated against the Disconnect-only delta, so the bound that
    #: divides by that cannot see the cascade constant being wrong. This
    #: denominator contains the unlisted bytes the calibration could not, and
    #: now that it is quiet enough to say so, the two agreeing at 1.07 and
    #: 1.05 is evidence about the unlisted half rather than a restatement of
    #: the calibration: if the cascade priced only the listed part of the
    #: subtree, this ratio would sit above the other rather than beside it.
    bytes_page_ratio_min_stable: float = 0.86
    bytes_page_ratio_max_stable: float = 1.27

    #: The same prediction over `MeasuredPageCost.bytes_saved_tracking_raw`:
    #: the two-arm delta counted only over Disconnect-matched requests.
    #: Measured 1.04.
    #:
    #: This is the module's sharp page-level bound, and everything page-level
    #: around it is weaker than it. What makes it sharp is the instrument
    #: rather than the estimator: restricting both arms to the requests the
    #: list names removes most of what makes the whole-page denominator move,
    #: because a page's two loads differ mostly in first-party media,
    #: carousels and lazy images.
    #:
    #: HOW WELL THE DENOMINATOR IS KNOWN. 5.1%, and that is measured rather
    #: than inferred, which is the change the ten-pass crawl bought. Each
    #: pass is an independent draw of the same experiment, so the spread
    #: across the ten *is* the sampling distribution: the Disconnect-only
    #: delta runs 45.3 to 71.6 MB a pass, a standard deviation of 16.2%, and
    #: the mean of ten therefore carries 5.1%. The whole-page delta over the
    #: same passes carries 23.4%, which is the one-line case for dividing by
    #: this and not that.
    #:
    #: The churn proxy `tracking_churn_se_pct` uses -- the untouched pages
    #: must shed nothing, so what they shed is noise -- was checked against
    #: that for the first time and came out well: over a single pass it
    #: predicts 16.4% where the passes measure 16.2%. That matters beyond
    #: this bound, because the per-category test still has to rely on the
    #: proxy at a grain the ten passes cannot speak to.
    #:
    #: SO THE BAND IS +-5%, one standard error of the denominator and ten of
    #: the numerator's, which at 0.5% is effectively fixed. It was +-10%, or
    #: two, and the same trade applies as for `bytes_page_ratio_*_stable`: a
    #: re-crawl moves this outside the band about a third of the time, and in
    #: exchange it catches a 5% scaling error where it needed 10%. This is
    #: the sharpest page-level bound in the module and now the one most
    #: likely to fail for reasons that are not the estimator's; read the
    #: per-pass spread in `pool_summary.json` before concluding anything from
    #: it. On the
    #: single-pass crawl this was +-7% around 1.47. Both the centre and the
    #: claim to precision were wrong there, and in the same way: one pass
    #: cannot see its own spread, and across the ten passes this ratio runs
    #: from 1.07 to 1.72. 1.47 was a draw from that, not a centre.
    #:
    #: WHAT IT NO LONGER SAYS. The centre used to derive rather than be
    #: observed, and that was the best thing about it: the estimator priced a
    #: blocked tracker's whole subtree at 70 KB a request while this
    #: denominator could only see the 36 KB of it that landed back on listed
    #: hosts, so 1.48 fell out of the arithmetic and 1.47 was measured. A
    #: drift toward 1.0 would have meant the estimator had stopped pricing
    #: the off-list half.
    #:
    #: It has now drifted to 1.0, and that is the intended consequence of a
    #: calibration rather than a regression. `FOLLOWUP_BYTES_PER_REQUEST` was
    #: 70 KB because the whole-page delta on one pass said so; on ten it says
    #: 26 KB, with an interval that includes both, and it returns a figure
    #: below the listed-only delta that is a subset of it. So the constant
    #: was re-set to the listed-only measurement, 47 KB, and both sides of
    #: this ratio are now the same quantity by construction.
    #:
    #: Be honest about what that costs: this bound is no longer independent
    #: of the estimator it grades. It is a calibration check -- does the
    #: shipped table still reproduce the crawl it was calibrated against --
    #: and it will not notice the cascade constant being wrong, because it is
    #: where the cascade constant came from. What it still catches is the
    #: direct table moving, which is two fifths of the numerator and was
    #: fitted on HTTP Archive rather than here, and anything that changes
    #: which requests get a cascade at all. For a statement that does not
    #: depend on this crawl being right, `external_ratio_*` is the only one
    #: in the module; for statements the aggregate calibration does not pin,
    #: the per-category, per-half and leave-one-out bounds below.
    #:
    #: Page-set variation is a separate thing and is not what this band
    #: covers: resampling which 271 pages you crawled moves this ratio over
    #: [0.78, 1.42]. That is what the subset bounds below are for -- and note
    #: that repeating the crawl does nothing for those, because they resample
    #: pages and the crawl has the same 271 it always had.
    bytes_tracking_ratio_min: float = 0.99
    bytes_tracking_ratio_max: float = 1.09

    #: How far each category's ratio may sit from the value recorded for it in
    #: `CATEGORY_TRACKING_RATIO`.
    #:
    #: The whole-crawl bound is one number over 271 pages, and a composition
    #: error cancels in it: an estimator 30% high on news and the same number
    #: of bytes low on productivity lands on the same 1.47 and passes -- which
    #: `COMPOSITION_SKEW` turns into a negative control rather than a claim.
    #: The per-request bounds are already cut by category for exactly that
    #: reason (`test_byte_bias_is_within_bounds_per_category`), and the
    #: page-level ones were not, because the whole-page delta has no power at
    #: that grain -- per category its churn standard error runs 45% to 150%
    #: and the measured ratios scatter over [0.22, 2.42], which is noise.
    #:
    #: The quiet instrument does have power there, on the two categories that
    #: carry the crawl: news at 5.6% churn standard error and productivity at
    #: 8.3%, together 426 MB of the 583 MB predicted. The rest sit at 22% to
    #: 49% and are excluded by measurement rather than by hand -- see
    #: `tracking_churn_se_pct` and `gradeable_tracking_se_pct`.
    #:
    #: This is one of the places the ten-pass crawl bought less than it looks
    #: like it should. The passes make each category's *delta* quieter, and
    #: the churn standard errors above are lower than the single-pass crawl's
    #: for that reason; what they do not change is that news is 39 pages and
    #: productivity 117, so a composition error still has to be large to
    #: stand clear of which pages happen to be in each bucket.
    #:
    #: A tolerance around each category's own measurement rather than one band
    #: around all of them. The two are much closer than they were -- 0.95 on
    #: news against 1.03 on productivity, where the single-pass crawl put
    #: them at 1.26 and 1.61 -- which is what recalibrating the cascade to
    #: this denominator does, and is not evidence that the categories agree.
    #: ±20% is 2.4 of productivity's standard errors and 3.6 of news's, and
    #: it catches a composition error of about a quarter or more;
    #: `COMPOSITION_SKEW` records where that line is by failing when the
    #: bound stops catching it.
    category_tracking_ratio_tolerance_pct: float = 20.0

    #: Churn standard error, as a share of a page group's Disconnect-only
    #: delta, above which the group is too noisy to bound. 15% sits in the
    #: gap the measurement leaves: news 5.6% and productivity 8.3% on one
    #: side, entertainment 22.0% and everything smaller on the other. The gap
    #: is wider on the pooled crawl than it was on one pass, and the same two
    #: categories fall on the same sides of it.
    gradeable_tracking_se_pct: float = 15.0

    #: The same ratio over *every* random half-crawl, not just their median.
    #: Measured over the 200 seeded halves: median 1.05, min 0.78, max 1.42.
    #:
    #: `subset_median_bytes_page_ratio_*` can only bound the median, because
    #: its denominator is a difference of two noisy whole-page totals and on
    #: an unlucky half it comes out near zero -- the ratio's 99th percentile
    #: across halves is 10.0 and the worst is 13.6. The quiet denominator does
    #: not do that: over the same 200 halves the ratio never leaves
    #: [0.78, 1.42], so the bound can be asserted on each half rather than on
    #: the middle of them, which is a much stronger statement than a bound on
    #: a median. Quarters, for the record, run [0.67, 1.67]: still finite,
    #: but at that size single pages start to carry a category.
    #:
    #: These are the bounds repeating the crawl did least for, and the spread
    #: here is why. A half-crawl is 136 of the same 271 pages; ten passes
    #: settle what each page sheds and change nothing about which pages there
    #: are. The width of [0.78, 1.42] is page-set variation, and the only
    #: thing that narrows it is crawling more domains.
    subset_bytes_tracking_ratio_min: float = 0.70
    subset_bytes_tracking_ratio_max: float = 1.55

    #: Cap on how far dropping any single page moves the whole-crawl
    #: Disconnect-only ratio. Measured +8.2%, and -1.3% the other way.
    #:
    #: The whole-page twin of this (`max_single_page_ratio_shift_pct`) has to
    #: tolerate 19.6%, so the quiet denominator is still the less
    #: concentrated of the two by a factor of more than two -- but it is more
    #: concentrated than it was, and the reason is one page rather than a
    #: trend. optimizely.com sheds 45.5 MB of listed bytes across ten passes,
    #: 8.1% of the denominator, against 60 blocked requests and 3.5 MB of
    #: estimate. Dropping it therefore moves the numerator hardly at all and
    #: the denominator a lot. It is a real page with a real saving, not an
    #: artefact, and the honest response is a bound that admits it rather
    #: than an exclusion list.
    max_single_page_tracking_ratio_shift_pct: float = 12.0

    #: Predicted bytes over the tracker mass HTTP Archive independently
    #: measured on the same domains, for the 181 pages it covers.
    #: Measured 0.20.
    #:
    #: The numerator is divided by `n_passes` first, because this is the one
    #: comparison whose denominator did not come from the pooled crawl:
    #: HTTP Archive measured each domain once. Everywhere else the factor
    #: cancels.
    #:
    #: The band is wide and the ratio is nowhere near 1, and neither is a
    #: defect: see `HTTP_ARCHIVE_CSV` for why the two quantities differ by a
    #: constant-ish factor. What this bound is for is independence. Every
    #: other number here comes from one crawl on one afternoon, so a
    #: systematic fault in it -- a misconfigured arm, a bad day on the
    #: network, a Playwright upgrade changing what loads -- would move the
    #: estimator's grade and every bound with it, invisibly. This denominator
    #: was measured by someone else, on different machines, in a different
    #: month, and it moves only when HTTP Archive is re-exported.
    #:
    #: Being loose is the price of that. It catches the estimator being wrong
    #: by a factor, not by a quarter, which is what the tight bounds above are
    #: for.
    external_ratio_min: float = 0.14
    external_ratio_max: float = 0.28

    # ----------------------------------------------------------------- #
    # Subsets
    # ----------------------------------------------------------------- #
    # Everything above is one number over the whole crawl, which a handful of
    # pages can carry. These bound the same quantities over `CATEGORIES` and
    # over `random_page_subsets`, so a pass has to hold up when the crawl is
    # cut apart. The bounds are wider than the whole-crawl ones by exactly as
    # much as the measured spread requires, and the negative controls in
    # `test_broken_estimates_are_detectably_bad` record what they still catch.

    #: Cap on the direct byte bias over a random half of the pages. Measured
    #: over 200 seeded halves: median +1.5%, 1st and 99th percentiles -8.6%
    #: and +14.5%, worst 18.4%. A half-crawl is a noisier instrument than the
    #: crawl, so the bar is looser than `max_abs_bytes_bias_pct` -- but it is
    #: applied to *every* subset, not on average.
    subset_max_abs_bytes_bias_pct: float = 25.0

    #: The same, over a random quarter. Measured worst 34.6%, p1/p99 -17.4%
    #: and +23.6%: at this size single pages start to dominate, so only the
    #: median is worth bounding tightly and the per-subset bar is loose.
    #:
    #: 80% on the single-pass crawl, where the worst quarter read 72.2%. This
    #: is the one subset bound repetition did narrow, and the mechanism is
    #: not page-set variation going away -- it is that each page's *observed*
    #: total is now ten loads rather than one, so a quarter-crawl carrying a
    #: handful of pages is carrying better-measured ones.
    quarter_subset_max_abs_bytes_bias_pct: float = 50.0

    #: Cap on |median bias| across the random halves. Measured +1.5%, close
    #: to the whole crawl's +2.1%, which is the point: halving the crawl
    #: should move the spread, not the centre.
    subset_median_abs_bytes_bias_pct: float = 8.0

    #: Band on the *median* page-level byte ratio across random halves, with
    #: follow-ups. Measured 1.28, close to the whole-crawl 1.34.
    #:
    #: Only the median is bounded tightly. The ratio's denominator is a
    #: difference of two noisy page totals, so on an unlucky half it can come
    #: out near zero and the ratio explodes: the 99th percentile across halves
    #: is 10.0 and the worst is 13.6. That is a property of the measurement,
    #: not of the estimator, and a test that pretended otherwise would just be
    #: a test of the seed. Those tails are three times what the single-pass
    #: crawl showed, for the arithmetic reason that the whole-page delta is
    #: now 39 MB a pass rather than 93 while its noise is not smaller: a
    #: denominator closer to zero explodes more easily.
    subset_median_bytes_page_ratio_min: float = 0.90
    subset_median_bytes_page_ratio_max: float = 1.75

    #: Share of random halves whose page-level ratio lands in a wide band.
    #: Measured 0.97 in [0.4, 4.0]. The band was [0.4, 2.5] when the same
    #: measurement read 0.98; it is the tails above that moved, not the
    #: estimator.
    subset_ratio_in_wide_band: float = 0.90
    wide_band_min: float = 0.4
    wide_band_max: float = 4.0

    #: Cap on how far dropping any single page moves the whole-crawl ratio.
    #: Measured -19.6%: tradingview.com's blocking arm transferred 106.0 MB
    #: *more* than its control arm across ten passes, so it is worth -27% of
    #: the whole-page denominator all by itself and dropping it lifts the
    #: denominator by more than a third. See `bytes_page_ratio_max_raw` for
    #: what that page is and why it is not excluded.
    max_single_page_ratio_shift_pct: float = 30.0


BOUNDS = Bounds()
