"""Ground truth from the paired top-500 Firefox crawl, for the estimate tests.

The journey tests measure the estimator against HTTP Archive, which is the
population it was fitted on. This module supplies the other kind of check: a
crawl of the Tranco top 500 where Firefox's own ETP decided what to block, and
the control arm recorded what those same requests really cost. Nothing here was
trained on.

Two ground truths come out of it, and they are not equally direct:

  bytes   Per request. The blocking arm names the requests Firefox refused;
          `compare_tracking_arms.py` matched them back to the control arm's
          `_transferSize`. Only the matched ones are usable, so this is the
          61.8% of blocked requests whose URL recurred across the two loads.
          Checked in aggregate, per page, and per stratum -- see
          `NEWS_MEDIA_DOMAINS` for why the aggregate alone is not enough.

  CPU     Per page only, because the crawl sampled the Firefox process tree
          rather than attributing cycles to individual requests. It also needs
          a drift correction: pages where ETP blocked nothing must show no
          saving, yet they total -28.3 MB of bytes and +16.7 s of CPU, which is
          crawl-to-crawl churn. Note the two run in opposite directions, so the
          correction raises the byte figure and lowers the CPU one.
          `load_measured_page_cost` subtracts both, per `src/eval_etp_proxy.py`.

Both tables are produced by:

    python src/firefox_crawl_500_tracking.py --mode normal  ...
    python src/firefox_crawl_500_tracking.py --mode private ...
    python src/compare_tracking_arms.py ...

and are gitignored, so `available()` reports a skip reason when they are absent.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CRAWL_DIR = ROOT / "data" / "raw" / "firefox_crawl_500_tracking"
BLOCKED_CSV = CRAWL_DIR / "blocked_observed_bytes.csv"
PAIRED_CSV = CRAWL_DIR / "paired_pages.csv"

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
            f"src/firefox_crawl_500_tracking.py and src/compare_tracking_arms.py "
            f"to run the top-500 estimate tests")


@dataclass(frozen=True)
class BlockedRequest:
    """One request Firefox refused, with what it really cost."""

    url: str
    resource_type: str
    observed_bytes: int


def load_blocked_requests() -> list[BlockedRequest]:
    """Tracking-protection blocks that could be priced against the control arm.

    Cryptomining and fingerprinting blocks are excluded: the crawl ran those
    protections in *both* arms, so they are not savings attributable to private
    browsing and the estimator is never asked about them here.

    This covers 1,162 of the 1,890 tracking blocks. The other 728 have no
    control-arm twin, so there is no per-request number to compare a prediction
    against -- the match *is* the measurement, and dropping the filter would
    only turn "unknown" into a spurious zero. They are not left ungraded,
    though: `test_blocked_bytes_do_not_exceed_page_level_saving` prices all
    1,890 against `MeasuredPageCost.bytes_saved`, which is measured at page
    level and so covers the unmatched ones too. That is a one-sided bound
    rather than a bias check, which is the most the data supports.
    """
    out: list[BlockedRequest] = []
    with open(BLOCKED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["protection"] != "tracking":
                continue
            # `observed_bytes` is the load-bearing test: a row has a priced
            # twin or it does not. `match_kind == "unmatched"` names exactly
            # the same rows -- `compare_tracking_arms.py` writes the two
            # together -- so it is not filtered on separately.
            if r["observed_bytes"] == "":
                continue
            out.append(BlockedRequest(
                url=r["blocked_url"],
                resource_type=r["resource_type"],
                observed_bytes=int(r["observed_bytes"]),
            ))
    return out


def load_all_tracking_blocks() -> list[BlockedRequest]:
    """Every tracking-protection block, whether or not it could be priced.

    The byte checks need a control-arm observation to compare against, so they
    use `load_blocked_requests`. A CPU estimate needs no such match -- the
    estimator prices a URL from the URL -- and the measured CPU it is compared
    against covers *all* the blocking, so restricting to the matched 61.8% here
    would compare a part against the whole. `observed_bytes` is -1 for the
    unpriced ones, which no caller should read.
    """
    out: list[BlockedRequest] = []
    with open(BLOCKED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["protection"] != "tracking":
                continue
            raw = r["observed_bytes"]
            out.append(BlockedRequest(
                url=r["blocked_url"],
                resource_type=r["resource_type"],
                observed_bytes=int(raw) if raw not in ("", None) else -1,
            ))
    return out


def load_blocked_by_page() -> dict[int, list[BlockedRequest]]:
    """The same requests, grouped by the page they were blocked on."""
    pages: dict[int, list[BlockedRequest]] = {}
    with open(BLOCKED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["protection"] != "tracking":
                continue
            # `observed_bytes` is the load-bearing test: a row has a priced
            # twin or it does not. `match_kind == "unmatched"` names exactly
            # the same rows -- `compare_tracking_arms.py` writes the two
            # together -- so it is not filtered on separately.
            if r["observed_bytes"] == "":
                continue
            pages.setdefault(int(r["page_idx"]), []).append(BlockedRequest(
                url=r["blocked_url"],
                resource_type=r["resource_type"],
                observed_bytes=int(r["observed_bytes"]),
            ))
    return pages


@dataclass(frozen=True)
class MeasuredPageCost:
    """What blocking measurably saved, in aggregate, across the crawl.

    `cpu_saved_s` and `bytes_saved` are drift-corrected: the pages ETP never
    touched are used to estimate crawl-to-crawl churn, and that estimate is
    subtracted from the pages it did touch.

    The two corrections move in opposite directions on this crawl, so neither
    raw total is safe to use: byte churn ran negative (-28.3 MB), which makes
    the raw 37.5 MB an understatement, while CPU churn ran positive (+16.7 s),
    which makes the raw CPU figure an overstatement.
    """

    n_pages_with_blocks: int
    n_pages_without_blocks: int
    cpu_saved_s: float
    bytes_saved: float
    cpu_drift_per_page_s: float


def load_measured_page_cost() -> MeasuredPageCost:
    with open(PAIRED_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["outcome_normal"].startswith("ok")
                and r["outcome_private"].startswith("ok")]

    def num(r: dict, key: str) -> float | None:
        v = r[key]
        return float(v) if v not in ("", None) else None

    blocked = [r for r in rows if int(r["n_blocked_tracking"]) > 0]
    untouched = [r for r in rows if int(r["n_blocked_tracking"]) == 0]

    cpu_blocked = [c for r in blocked if (c := num(r, "cpu_s_saved")) is not None]
    cpu_untouched = [c for r in untouched
                     if (c := num(r, "cpu_s_saved")) is not None]
    cpu_drift = st.fmean(cpu_untouched) if cpu_untouched else 0.0

    bytes_blocked = sum(int(r["bytes_saved"]) for r in blocked)
    bytes_drift = (st.fmean([int(r["bytes_saved"]) for r in untouched])
                   if untouched else 0.0)

    return MeasuredPageCost(
        n_pages_with_blocks=len(blocked),
        n_pages_without_blocks=len(untouched),
        cpu_saved_s=sum(cpu_blocked) - cpu_drift * len(cpu_blocked),
        bytes_saved=bytes_blocked - bytes_drift * len(blocked),
        cpu_drift_per_page_s=cpu_drift,
    )


# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #
# Ad-supported news, magazine and general-interest media publishers among the
# crawl's pages. The aggregate byte bound is a sum over all 500 domains, so a
# bias confined to one kind of page cancels against the rest and passes, which
# is what this stratum exists to catch. It did: the split read news -18.7%
# against other +0.1% while the aggregate sat at -7.1%, comfortably inside the
# bound. After the host rungs it reads news -8.5%, other +1.1%, aggregate
# -2.6% -- so keep the stratum, since it is the only thing here that would see
# the next such bias.
#
# Membership was assigned from the domain name alone, on editorial grounds,
# and deliberately not from prediction error -- a stratum drawn around the
# pages that happen to score badly would guarantee its own result. Pure
# platforms are excluded even when ad-supported and content-heavy
# (substack.com, imgur.com, dailymotion.com, scribd.com, wattpad.com,
# twitter.com), because the page being priced is a feed or a viewer rather
# than an article. Portals are included where news is the landing page's
# substance (163.com, aol.com, mail.ru, dzen.ru, onet.pl, seznam.cz,
# sina.com.cn, sohu.com, uol.com.br).
#
# Registrable domains, matched against the page URL's host.
NEWS_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "163.com", "aol.com", "bbc.co.uk", "bbc.com", "cnet.com", "cnn.com",
    "dailymail.co.uk", "dailymail.com", "dzen.ru", "elpais.com", "espn.com",
    "foxnews.com", "globo.com", "hbr.org", "independent.co.uk",
    "indiatimes.com", "latimes.com", "lefigaro.fr", "lemonde.fr", "mail.ru",
    "mlb.com", "nbcnews.com", "nikkei.com", "npr.org", "nytimes.com",
    "onet.pl", "people.com", "seznam.cz", "sina.com.cn", "sohu.com",
    "techcrunch.com", "theverge.com", "time.com", "uol.com.br", "weather.com",
    "welt.de",
})

STRATA: tuple[str, ...] = ("news_media", "other")


def stratum_for(page_url: str) -> str:
    """Which stratum a crawled page belongs to."""
    host = urlparse(page_url).netloc or page_url
    host = host.split("@")[-1].split(":")[0].removeprefix("www.").lower()
    return "news_media" if host in NEWS_MEDIA_DOMAINS else "other"


def load_blocked_by_stratum() -> dict[str, list[BlockedRequest]]:
    """Priced tracking blocks, grouped by the stratum of the page they are on."""
    out: dict[str, list[BlockedRequest]] = {k: [] for k in STRATA}
    with open(BLOCKED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["protection"] != "tracking":
                continue
            if r["observed_bytes"] == "":
                continue
            out[stratum_for(r["page_url"])].append(BlockedRequest(
                url=r["blocked_url"],
                resource_type=r["resource_type"],
                observed_bytes=int(r["observed_bytes"]),
            ))
    return out


# A stratum that misses `Bounds.max_abs_bytes_bias_pct` must be recorded here
# with its measured bias, exactly as `solutions.KNOWN_UNCALIBRATED` records a
# miss on the journey suite, and only the strata in `BIASED_STRATA_SKIPPED` are
# let off with a skip. Widening the bound to fit a stratum would defeat the
# point of adding it.
KNOWN_BIASED_STRATA: dict[str, str] = {}

#: Strata whose miss is recorded rather than failing the suite.
#:
#: Empty. `news_media` was here at -18.7%, and the fix was the host rungs in
#: `llm-classifier`: the stratum's gap was almost entirely ad and
#: content-recommendation *scripts*, whose vendors serve versioned bundles from
#: per-customer subdomains, so every one of them missed the path rungs and was
#: answered with the mean of any .js (37 KB) instead of its own host's (up to
#: 179 KB). It now measures -8.5%, inside the bound, with `other` unmoved at
#: +1.1%. See `Level::Host` in llm-classifier/src/lib.rs.
BIASED_STRATA_SKIPPED: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def predict(requests: list[BlockedRequest]) -> tuple[list[int], list[float]]:
    """(bytes, cpu_ms) from the shipped table, per request.

    The estimator sees only the URL and the request context -- what Firefox has
    before a response exists -- so nothing here leaks the observed size.

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
            llm_classifier.RequestInitiator.UNKNOWN, "")
        sizes.append(b)
        cpu.append(ms)
    return sizes, cpu


# Deliberately wrong predictors, as a negative control on the bounds below. If
# a bug made the comparison insensitive to prediction quality, these would pass
# it, so the tests assert that they do not.
BROKEN_SCALES: dict[str, float] = {
    "zeroed": 0.0,
    "tenfold": 10.0,
    "tenth": 0.1,
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
    #: Measured -2.6%; the cap matches the journey suite's MAX_SIGNED_BIAS_PCT.
    max_abs_bytes_bias_pct: float = 12.0

    #: Floor on the share of pages whose predicted byte total lands within 25%
    #: of the observed total. Measured 0.459.
    min_pages_within_25pct: float = 0.35

    #: Predicted CPU over *all* tracking blocks, as a fraction of the CPU
    #: measured at page level. Measured 0.54.
    #:
    #: The ceiling is the interesting half and is 1.0 on principle rather than
    #: from the measurement: a per-request estimate must not exceed the
    #: page-level saving, because blocking a tracker also prevents the
    #: subresources it would have requested and executed, and the estimator does
    #: not model those. Predicting more CPU than was measurably saved would mean
    #: it is over-attributing.
    #:
    #: The floor guards the other way -- against the coefficients collapsing to
    #: nothing, which would otherwise look like a pass.
    #:
    #: Be clear about the sensitivity this buys: with the measurement at 0.54,
    #: the band tolerates scaling every CPU estimate by anything from 0.46x to
    #: 1.9x. It catches the model breaking, not the coefficients being somewhat
    #: wrong -- and it cannot do better while `cpu_ms` is derived rather than
    #: fitted and the only ground truth is per page. The negative control in
    #: `test_broken_estimates_are_detectably_bad` records where the line is.
    cpu_ratio_min: float = 0.25
    cpu_ratio_max: float = 1.0

    #: Predicted bytes over *all* tracking blocks, as a fraction of the bytes
    #: measured saved at page level. Measured 0.26.
    #:
    #: This is the byte twin of the CPU band, and exists to say something about
    #: the 728 blocks that `load_blocked_requests` cannot price. The ceiling is
    #: the same structural 1.0: blocking a tracker also prevents whatever it
    #: would have gone on to request, so the page-level saving contains bytes
    #: the estimator never claims, and predicting more than was measured means
    #: over-attribution.
    #:
    #: Read the floor with care. It sits far below the ceiling because the
    #: cascade is most of the effect -- 26.1 MB predicted against 100.1 MB
    #: measured -- so this bound cannot detect bias, only a collapse or a gross
    #: over-claim. The per-request check in
    #: `test_blocked_byte_total_is_within_bounds` is the sharp one, and it is
    #: still restricted to the matched 61.5%.
    bytes_page_ratio_min: float = 0.10
    bytes_page_ratio_max: float = 1.0


BOUNDS = Bounds()
