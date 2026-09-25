"""Are the size and CPU estimates for the top-500 crawl inside their bounds?

The journey suite grades the estimator against HTTP Archive, the log it was
fitted on. These tests grade it against the paired Tranco top-500 Firefox
crawl instead, where ETP itself chose what to block and the control arm
recorded what those requests really cost. It is out-of-sample in every sense
that matters: different pages, different crawler, and a blocking decision made
by the browser rather than by a list lookup.

Bytes are checked per request and per page over the blocks that could be
priced, plus once across the whole blocked population against the page-level
saving; CPU only in aggregate, because the
crawl sampled the Firefox process tree per page and never attributed cycles to
individual requests. The CPU bound is therefore a much weaker statement than
the byte bound, which is the honest reflection of the fact that `cpu_ms` is
derived from the byte estimate rather than fitted -- see `cpu_ms_for` in
`llm-classifier/src/lib.rs`.

Bounds live in `top500.Bounds` with their measured values recorded beside them.
The crawl tables are gitignored, so every test here skips with an explanatory
message when they are absent.
"""

from __future__ import annotations

import statistics
from urllib.parse import urlparse

import pytest

import top500


@pytest.fixture(scope="module", autouse=True)
def _require_crawl() -> None:
    """Skip the module when the crawl output has not been generated."""
    unavailable = top500.available()
    if unavailable:
        pytest.skip(unavailable, allow_module_level=True)


@pytest.fixture(scope="module")
def blocked() -> list[top500.BlockedRequest]:
    requests = top500.load_blocked_requests()
    assert requests, "no priced tracking blocks in the crawl output"
    return requests


@pytest.fixture(scope="module")
def predicted(blocked: list[top500.BlockedRequest]) -> tuple[list[int], list[float]]:
    return top500.predict(blocked)


@pytest.fixture(scope="module")
def all_blocked() -> list[top500.BlockedRequest]:
    """Every tracking block, including those with no priced control-arm twin.

    The CPU comparison is against CPU measured over all the blocking, so the
    estimate has to cover all of it too.
    """
    requests = top500.load_all_tracking_blocks()
    assert requests, "no tracking blocks in the crawl output"
    return requests


@pytest.fixture(scope="module")
def predicted_cpu_s(all_blocked: list[top500.BlockedRequest]) -> float:
    _sizes, cpu_ms = top500.predict(all_blocked)
    return sum(cpu_ms) / 1000.0


@pytest.fixture(scope="module")
def measured() -> top500.MeasuredPageCost:
    return top500.load_measured_page_cost()


def test_blocked_byte_total_is_within_bounds(
        blocked: list[top500.BlockedRequest],
        predicted: tuple[list[int], list[float]]) -> None:
    """The estimated total for what ETP blocked tracks the observed total.

    This is the dashboard's actual claim -- a sum, not a per-request figure --
    restated against a browser's own blocking decisions.
    """
    sizes, _cpu = predicted
    observed = sum(r.observed_bytes for r in blocked)
    assert observed > 0, "observed bytes sum to zero; the ground truth is empty"

    bias_pct = 100.0 * (sum(sizes) - observed) / observed
    assert abs(bias_pct) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
        f"top-500 blocked bytes estimated {bias_pct:+.1f}% off "
        f"({sum(sizes)/1e6:.1f} MB predicted vs {observed/1e6:.1f} MB observed) "
        f"over {len(blocked):,} requests, outside the "
        f"{top500.BOUNDS.max_abs_bytes_bias_pct:.0f}% bound"
    )


@pytest.mark.parametrize("category", top500.CATEGORIES)
def test_byte_bias_is_within_bounds_per_category(category: str) -> None:
    """Each kind of page is unbiased on its own, not just the crawl as a whole.

    `test_blocked_byte_total_is_within_bounds` sums over all 500 domains, so a
    bias confined to one kind of page cancels against the others and passes.
    It has twice. News and media pages once ran ~19% low while the rest of the
    crawl ran +0.1% and the total landed at -7.1%, comfortably inside the
    bound; the host rungs in `llm-classifier` closed that to -6.5%. And
    `infrastructure` runs +33.7% today against an aggregate of +2.1% -- see
    `KNOWN_BIASED_CATEGORIES` for what that is and why it is recorded rather
    than fixed. Both are cases the aggregate could not see.

    The bar is the same on every category on purpose -- there is no reason a
    news page should be allowed more error than a bank's. A category that
    misses it must be recorded in `KNOWN_BIASED_CATEGORIES` with its measured
    bias, and only the ones in `BIASED_CATEGORIES_SKIPPED` are let off with a
    skip; any other miss is a hard failure. So a category that goes bad later
    fails loudly rather than passing a threshold widened to fit it.

    Measured: entertainment -11.0%, infrastructure +33.7%, news -1.5%,
    productivity +4.5%, reference +2.1%, shopping -6.8%, social -2.7%.
    """
    requests = top500.load_blocked_by_category()[category]
    assert requests, f"category {category!r} has no priced blocked requests"
    observed = sum(r.observed_bytes for r in requests)
    assert observed > 0, f"category {category!r} has no observed bytes"

    sizes, _cpu = top500.predict(requests)
    bias_pct = 100.0 * (sum(sizes) - observed) / observed

    known = top500.KNOWN_BIASED_CATEGORIES.get(category)
    if known is not None and abs(bias_pct) > top500.BOUNDS.max_abs_bytes_bias_pct:
        if category in top500.BIASED_CATEGORIES_SKIPPED:
            pytest.skip(f"{category} is known to be biased: {known} "
                        f"(measured {bias_pct:+.1f}%), skipping")
        pytest.fail(f"{known} (measured {bias_pct:+.1f}%)")

    assert abs(bias_pct) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
        f"{category}: blocked bytes estimated {bias_pct:+.1f}% off "
        f"({sum(sizes)/1e6:.1f} MB predicted vs {observed/1e6:.1f} MB observed) "
        f"over {len(requests):,} requests on this category alone, outside the "
        f"{top500.BOUNDS.max_abs_bytes_bias_pct:.0f}% bound -- the aggregate "
        f"test can miss this when other categories cancel it"
    )


def test_category_bias_is_not_hidden_by_the_aggregate() -> None:
    """The split must be able to see what the aggregate cannot.

    If the categories ever collapsed -- one emptied by a loader change, or
    every page landing in `infrastructure` because host matching broke -- the
    test above would still pass while measuring nothing. This asserts the
    split is real and that it is actually sharper than the grand total.
    """
    by_category = top500.load_blocked_by_category()
    biases = {}
    for name, requests in by_category.items():
        observed = sum(r.observed_bytes for r in requests)
        assert len(requests) >= 50, (
            f"category {name!r} has only {len(requests)} priced requests, too "
            f"few to bound a bias against"
        )
        sizes, _cpu = top500.predict(requests)
        biases[name] = 100.0 * (sum(sizes) - observed) / observed

    all_requests = top500.load_blocked_requests()
    all_observed = sum(r.observed_bytes for r in all_requests)
    all_sizes, _cpu = top500.predict(all_requests)
    aggregate = 100.0 * (sum(all_sizes) - all_observed) / all_observed

    worst = max(abs(b) for b in biases.values())
    assert worst > abs(aggregate), (
        f"no category is further off than the {aggregate:+.1f}% aggregate "
        f"({', '.join(f'{k} {v:+.1f}%' for k, v in biases.items())}), so "
        f"splitting is measuring nothing the grand total did not already show"
    )


# --------------------------------------------------------------------------- #
# Tracker roles
# --------------------------------------------------------------------------- #
# Everything above cuts the crawl by the page a request was blocked on. These
# cut it by the request: what kind of tracker it was, as `family-kind`, from
# the Disconnect list's own category and the request context. See
# `top500.TRACKER_FAMILIES` for the taxonomy, for why both halves of it are
# measured rather than invented, and for the functional taxonomy that was
# tried first and does not survive the data.


def test_no_tracker_role_dominates_the_crawl_bias() -> None:
    """No one kind of tracker may push the crawl's total around on its own.

    The bound this axis exists for. The aggregate byte bias is +2.1%, which
    reads like accuracy and is not: it is the sum of six roles contributing
    -1.9, +1.4, +2.7, +1.7, -2.0 and +0.8 points of the crawl's observed
    bytes. Take the absolute values and the estimator is 11 points wrong by
    role and 2 points wrong in total, so nine tenths of its error is
    cancellation between kinds of tracker.

    Cancellation is not a defect by itself -- a conditional mean is supposed
    to be wrong in both directions -- but it is invisible, and what is
    invisible can move. Every bound that existed before this one is either a
    total, which cancellation is defined to survive, or a cut by page, which
    does not separate roles because every kind of page carries every kind of
    tracker. `test_a_tracker_composition_error_is_invisible_to_the_page_split`
    is the demonstration rather than the claim.

    Bounded on the contribution rather than the relative bias on purpose;
    `Bounds.max_tracker_role_bias_contribution_pct` says why, and the short
    version is that what the estimator promises is an unbiased total, so the
    thing to bound is what each role does to the total.
    """
    by_role = top500.load_blocked_by_tracker_role()
    total_observed = sum(r.observed_bytes
                         for rs in by_role.values() for r in rs)
    assert total_observed > 0, "no observed bytes to bound against"

    contributions = {}
    for role, requests in by_role.items():
        sizes, _cpu = top500.predict(requests)
        observed = sum(r.observed_bytes for r in requests)
        contributions[role] = 100.0 * (sum(sizes) - observed) / total_observed

    worst = max(contributions, key=lambda k: abs(contributions[k]))
    cap = top500.BOUNDS.max_tracker_role_bias_contribution_pct
    assert abs(contributions[worst]) <= cap, (
        f"{worst} alone moves the crawl's byte total by "
        f"{contributions[worst]:+.1f}% of the {total_observed/1e6:.1f} MB "
        f"observed, outside the {cap:.0f}% cap "
        f"({', '.join(f'{k} {v:+.1f}' for k, v in sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:5])}) "
        f"-- a composition error across kinds of tracker, which the "
        f"aggregate and the page split both cancel"
    )


def test_byte_bias_is_within_bounds_per_tracker_role() -> None:
    """Each kind of tracker is unbiased on its own, where the crawl can say.

    The relative-bias twin of the contribution bound above, and the twin of
    `test_byte_bias_is_within_bounds_per_category` on the other axis. Same
    bar as everywhere else -- there is no reason an ad loader should be
    allowed more error than an analytics one.

    Only roles that clear both floors in
    `Bounds.gradeable_tracker_role_byte_share` and
    `..._requests` are graded, and which those are is itself asserted, so a
    role growing into gradeability or falling out of it fails here rather
    than changing what the suite measures in silence.

    Measured: ad-script -3.6%, analytics-script +6.9%, social-script +16.8%,
    ad-beacon +41.6%. The last two miss and are recorded in
    `top500.KNOWN_BIASED_TRACKER_ROLES` with their values, which this checks
    rather than merely consults -- a recorded miss that drifts is a change
    worth failing on.
    """
    by_role = top500.load_blocked_by_tracker_role()
    total_observed = sum(r.observed_bytes
                         for rs in by_role.values() for r in rs)

    graded, biases = {}, {}
    for role, requests in by_role.items():
        observed = sum(r.observed_bytes for r in requests)
        if observed <= 0:
            continue
        sizes, _cpu = top500.predict(requests)
        biases[role] = 100.0 * (sum(sizes) - observed) / observed
        if (observed / total_observed
                >= top500.BOUNDS.gradeable_tracker_role_byte_share
                and len(requests)
                >= top500.BOUNDS.gradeable_tracker_role_requests):
            graded[role] = biases[role]

    assert graded, "no tracker role is large enough to grade"
    expected = {"ad-script", "analytics-script"} | set(
        top500.KNOWN_BIASED_TRACKER_ROLES)
    assert set(graded) == expected, (
        f"the gradeable roles are {sorted(graded)} but {sorted(expected)} "
        f"were expected ({', '.join(f'{k} {v:+.1f}%' for k, v in sorted(biases.items()))}) "
        f"-- a role has crossed a floor in `Bounds.gradeable_tracker_role_*`; "
        f"measure it and record it, or drop it, but do not let the set move "
        f"on its own"
    )

    tolerance = top500.BOUNDS.known_tracker_role_bias_tolerance_pct
    for role, bias in sorted(graded.items()):
        recorded = top500.KNOWN_BIASED_TRACKER_ROLES.get(role)
        if recorded is not None:
            off = 100.0 * (bias - recorded) / abs(recorded)
            assert abs(off) <= tolerance, (
                f"{role} is {bias:+.1f}% off, against the {recorded:+.1f}% "
                f"recorded for it in KNOWN_BIASED_TRACKER_ROLES -- a drift of "
                f"{off:+.0f}%, outside the {tolerance:.0f}% this registry "
                f"allows. A miss that is recorded rather than fixed still has "
                f"to stay where it was recorded"
            )
            continue
        assert abs(bias) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
            f"{role}: blocked bytes estimated {bias:+.1f}% off over "
            f"{len(by_role[role]):,} priced requests, outside the "
            f"{top500.BOUNDS.max_abs_bytes_bias_pct:.0f}% bound -- either fix "
            f"it or record it in KNOWN_BIASED_TRACKER_ROLES with its value "
            f"and what it is"
        )


def test_tracker_roles_cover_the_crawl() -> None:
    """The taxonomy names what the crawl actually contains.

    `tracker_role_for` falls back to `other` on both halves, which is the
    right behaviour for a Disconnect list that gains a category or a crawler
    that reports a context this module has not seen, and the wrong thing to
    discover silently: mass drifting into `other-*` would quietly empty the
    roles the bounds above are computed over. So the fallback is bounded.

    Measured: `other-script` is 20 of 11,290 priced requests and 0.4% of
    their bytes, all of them URLs the shipped Disconnect list no longer names
    but the crawl's Firefox still blocked.
    """
    by_role = top500.load_blocked_by_tracker_role()
    total = sum(r.observed_bytes for rs in by_role.values() for r in rs)
    unnamed = sum(r.observed_bytes for role, rs in by_role.items()
                  for r in rs if role.startswith("other-")
                  or role.endswith("-other"))
    assert unnamed <= 0.05 * total, (
        f"{unnamed/1e6:.1f} MB of {total/1e6:.1f} MB ({unnamed/total:.1%}) "
        f"falls in a fallback role, over 5% -- the Disconnect list or the "
        f"crawler's resource types have moved under "
        f"`top500.TRACKER_FAMILIES`/`TRACKER_KINDS`, and the per-role bounds "
        f"are grading less of the crawl than they appear to"
    )


def test_a_tracker_composition_error_is_invisible_to_the_page_split() -> None:
    """Negative control: why this axis is not the page split again.

    A cut only earns its place if it sees something the existing cuts do not.
    `top500.TRACKER_COMPOSITION_SKEW` lifts `ad-script` by 15% and drops
    `analytics-script` by the same number of bytes -- an estimator that
    over-prices ad loaders and under-prices analytics ones by the same mass,
    which is a specific and plausible way to be wrong.

    This asserts all three halves of the claim. The aggregate must still
    pass, because a total cannot see a reallocation. *Every page category*
    must still pass, which is the part that is not obvious and is the whole
    argument for the axis: both roles appear on every kind of page, so moving
    bytes between them barely moves any page's bias. And the contribution
    bound must fail.
    """
    role_a, role_b, skew = top500.TRACKER_COMPOSITION_SKEW
    requests = top500.load_blocked_requests()
    roles = [top500.tracker_role_for(r.url, r.resource_type) for r in requests]
    sizes, _cpu = top500.predict(requests)
    observed = [r.observed_bytes for r in requests]
    total = sum(observed)

    in_a = sum(s for s, role in zip(sizes, roles) if role == role_a)
    in_b = sum(s for s, role in zip(sizes, roles) if role == role_b)
    assert in_a > 0 and in_b > 0, (
        f"{role_a} or {role_b} has no predicted bytes, so the skew cannot be "
        f"applied -- has the Disconnect list changed under the taxonomy?"
    )
    moved = skew * in_a
    factor = {role_a: 1.0 + skew, role_b: 1.0 - moved / in_b}
    skewed = [s * factor.get(role, 1.0) for s, role in zip(sizes, roles)]

    aggregate = 100.0 * (sum(skewed) - total) / total
    assert abs(aggregate) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
        f"moving {moved/1e6:.1f} MB from {role_b} to {role_a} changed the "
        f"aggregate bias to {aggregate:+.1f}%, so this control is no longer "
        f"testing what a total cannot see -- pick a skew that cancels"
    )

    for category in top500.CATEGORIES:
        if category in top500.BIASED_CATEGORIES_SKIPPED:
            continue
        idx = [i for i, r in enumerate(requests) if r.category == category]
        if not idx:
            continue
        obs = sum(observed[i] for i in idx)
        bias = 100.0 * (sum(skewed[i] for i in idx) - obs) / obs
        assert abs(bias) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
            f"the skew takes page category {category} to {bias:+.1f}%, which "
            f"the page split would catch -- so it no longer demonstrates that "
            f"the tracker axis sees something the page axis cannot. Shrink "
            f"`TRACKER_COMPOSITION_SKEW`"
        )

    worst = 0.0
    for role in (role_a, role_b):
        idx = [i for i, x in enumerate(roles) if x == role]
        obs = sum(observed[i] for i in idx)
        contribution = 100.0 * (sum(skewed[i] for i in idx) - obs) / total
        worst = max(worst, abs(contribution))
    assert worst > top500.BOUNDS.max_tracker_role_bias_contribution_pct, (
        f"the skewed roles contribute at most {worst:.1f}% of the crawl's "
        f"bytes in error, inside the "
        f"{top500.BOUNDS.max_tracker_role_bias_contribution_pct:.0f}% cap, so "
        f"`test_no_tracker_role_dominates_the_crawl_bias` would pass a "
        f"composition error every other bound here is blind to"
    )


def test_per_page_byte_totals_are_within_bounds() -> None:
    """Most pages' predicted totals land near the truth, not just the grand sum.

    A grand total can be right because two large errors cancelled. The
    dashboard rolls up a user's own browsing, so the per-page distribution is
    what they would actually see.
    """
    by_page = top500.load_blocked_by_page()
    ratios = []
    for requests in by_page.values():
        observed = sum(r.observed_bytes for r in requests)
        if observed <= 0:
            continue  # all-beacon page: a ratio against zero says nothing
        sizes, _cpu = top500.predict(requests)
        ratios.append(sum(sizes) / observed)

    assert ratios, "no pages with non-zero observed blocked bytes"
    within = sum(1 for r in ratios if 0.75 <= r <= 1.25) / len(ratios)
    assert within >= top500.BOUNDS.min_pages_within_25pct, (
        f"only {within:.1%} of {len(ratios)} pages land within 25% of their "
        f"observed blocked-byte total (needs "
        f"{top500.BOUNDS.min_pages_within_25pct:.0%})"
    )


def test_blocked_bytes_do_not_exceed_page_level_saving(
        all_blocked: list[top500.BlockedRequest],
        measured: top500.MeasuredPageCost) -> None:
    """Price *every* tracking block, including the 7,459 with no priced twin.

    `test_blocked_byte_total_is_within_bounds` is the sharper check but it can
    only grade the 60.2% of blocks whose URL recurred across the two arms; the
    rest have no per-request observation to compare against. This one covers
    all 18,749 by comparing against the page-level byte saving, the same way
    the CPU tests do, so a regression confined to the unmatched population
    cannot pass unnoticed.

    Scored with `include_followups`, because the denominator has them in it:
    the page-level saving is everything the page did not fetch, subresources
    a blocked tracker would have gone on to request included. Priced per
    request and directly, the estimate comes to 0.57x of it; with the cascade
    priced too, 1.34x.

    Both the band and the argument behind it changed when the crawl grew from
    one pass to ten -- the ceiling here is no longer structural, because the
    whole-page delta turns out not to bound what blocking removed. See
    `Bounds.bytes_page_ratio_min` for the re-derivation and for what the band
    can and cannot detect, and `FOLLOWUP_BYTES_PER_REQUEST` in
    `llm-classifier/src/lib.rs` for how roughly the cascade is known.
    """
    sizes, _cpu = top500.predict(all_blocked, include_followups=True)
    predicted = sum(sizes)
    assert measured.bytes_saved > 0, (
        f"measured byte saving is not positive ({measured.bytes_saved/1e6:.1f} MB); "
        f"the crawl cannot bound the byte estimate"
    )

    ratio = predicted / measured.bytes_saved
    assert (top500.BOUNDS.bytes_page_ratio_min <= ratio
            <= top500.BOUNDS.bytes_page_ratio_max), (
        f"predicted {predicted/1e6:.1f} MB for {len(all_blocked):,} blocked "
        f"requests is {ratio:.2f}x the {measured.bytes_saved/1e6:.1f} MB measured "
        f"across {measured.n_pages_with_blocks} pages, outside "
        f"[{top500.BOUNDS.bytes_page_ratio_min}, "
        f"{top500.BOUNDS.bytes_page_ratio_max}]"
    )


def test_blocked_bytes_are_within_bounds_against_the_raw_page_delta(
        all_blocked: list[top500.BlockedRequest],
        measured: top500.MeasuredPageCost) -> None:
    """The same ratio against the delta as measured, with no drift subtracted.

    `test_blocked_bytes_do_not_exceed_page_level_saving` divides by
    `bytes_saved`, which this module drift-corrects, so a correction derived
    here is part of what decides whether it passes. This one divides by what
    the two arms actually differed by, so the estimator has to clear the bar
    without that help. On this crawl the distinction earns its keep twice
    over: the drift correction is worth +42 MB on the whole-page denominator
    and +24 MB on the Disconnect-only one, and it is also the *noisier*
    choice -- correcting the quiet denominator takes its standard error from
    5.1% to 9.2%, because the untouched pages' churn has to be scaled up by
    271/199 to apply it and that amplifies its own error. The raw delta is
    both the more honest and the better-measured instrument here.

    It does that three times, over the same prediction and three versions of
    the same delta, and the three assertions are no longer the same test at
    three widths. Two of them were, on the single-pass crawl: the
    denominators stood in a fixed ratio of 1.753 and the whole-page band
    strictly contained the tracking one, so the second always bound the
    first. That is no longer so, and the third did not exist.

    Over every byte the 271 pages shed: 1.48x of 392.8 MB, inside [1.05,
    1.91] -- one jackknifed standard error of the denominator either way, and
    see `Bounds.bytes_page_ratio_max_raw` for what a one-sigma band costs.
    This used to be the module's structural assertion, on the argument
    that claiming more than the two arms literally differed by is
    over-attribution under any reading of the crawl. Pooling ten passes
    retired that argument, because the Disconnect-only delta -- a strict
    subset of this one, the same difference over fewer requests -- came out
    42% *larger*, 559.4 MB against 392.8, which can only mean the blocking
    arm fetches more of something no list names. The band was widened to a
    factor-level sanity check around 1.48 and left there.

    That was the wrong conclusion, and the crawl's own passes say so. It is
    kept as written above because a bound stated over every page the crawl
    touched needs no argument, and the assertion that follows it is the one
    to read.

    Over every byte the 265 pages *whose loads repeat* shed: 1.07x of
    542.2 MB, inside [0.86, 1.27]. Six pages score above 1.0 on
    `top500.page_delta_instability` -- their two arms differ, pass to pass,
    by about as much as the page weighs -- and they carry -149.4 MB of delta
    against 3.9 MB of predicted saving. Drop them and the whole-page delta
    goes from 392.8 MB to 542.2 MB, the ratio from 1.48 to 1.07, and the
    denominator's standard error from 23.4% to 16.2%. The 166.6 MB of
    unlisted fetching at t = -2.1 was tradingview.com, a streaming charting
    app, and five others; the subset is back below its superset, 542.2
    against 549.7, a 1.4% gap on an instrument that carries 5.1%.

    Two things make that a bound rather than a page-drop that flatters the
    estimator. The score never reads the level of a page's delta, only its
    spread and the page's size, so no page can be cut for disagreeing. And
    the quiet instrument is the control: over the same six pages it moves
    -1.7% where the loud one moves +38%, which is what separates removing
    churn from removing signal. `Bounds.bytes_page_ratio_max_stable` has the
    rest, including why 1.0 and why anything from 0.5 to 2.0 would do.

    This is also the answer to the note above that repeating the crawl bought
    nothing at page level. Ten passes summed into one total cannot remove a
    page whose loads do not repeat -- they average it in ten times. Ten
    passes read one page at a time can, and nothing else can, which makes
    this the only page-level bound in the module a single-pass crawl could
    not have stated.

    Over the bytes of Disconnect-matched requests alone: 1.04x of 559.4 MB,
    bounded to ±5%, and this is the module's sharp page-level bound. What
    makes it sharp is the instrument, not the estimator: restricting both
    arms to the requests the list names removes most of what makes the first
    denominator noisy. The ten passes say how much, by measurement rather
    than by proxy -- each pass is a draw of the same experiment, and across
    them the Disconnect-only delta has a standard deviation of 16.2% against
    the whole page's 74%, so the mean of ten carries 5.1% against 23.4%.

    The centre is 1.04 and, unlike the 1.47 this replaced, it does not derive
    from anything. That is a real loss and it is worth being plain about.
    The old centre fell out of the arithmetic: the estimator priced a blocked
    tracker's whole subtree at 70 KB a request, this denominator could only
    see the 36 KB of it that landed on listed hosts, and 70/36 through the
    direct estimate gave 1.48 against 1.47 measured. `FOLLOWUP_BYTES_PER_REQUEST`
    has since been recalibrated to this crawl's listed-only figure, 47 KB,
    because the whole-page instrument that justified 70 returns 26 KB on ten
    passes and a value below a subset of itself. Both sides of this ratio are
    now the same quantity, so a pass says the shipped table still reproduces
    the crawl it was calibrated against -- a calibration check, not an
    independent bound. `test_blocked_bytes_are_within_bounds_against_http_archive`
    is the one statement here that does not depend on this crawl at all.

    The stable whole-page assertion recovers part of what that costs. The
    calibration was fitted against the listed-only delta and so cannot see
    the cascade constant being wrong; the whole-page delta contains the
    unlisted bytes the calibration could not, and it is now quiet enough to
    be read. The two landing together, 1.07 and 1.05, is therefore evidence
    about the unlisted half of the cascade rather than a restatement of the
    fit: price only the listed part of a blocked tracker's subtree and this
    ratio would sit above the other, not beside it.

    A failure is still not automatically the estimator's fault, and the three
    bounds below are how to tell: they cut the same ratio by category, by
    random half and by single page, and none of them is pinned by the
    aggregate calibration. Which assertion fails narrows it further. The
    first alone is churn: read `n_pages_without_blocks` and the drift figure.
    The first and second together, with the third passing, is a page the
    filter should have caught and did not. All three is the estimator.
    """
    sizes, _cpu = top500.predict(all_blocked, include_followups=True)
    predicted = sum(sizes)
    assert measured.bytes_saved_raw > 0, (
        f"raw measured byte saving is not positive "
        f"({measured.bytes_saved_raw/1e6:.1f} MB) across "
        f"{measured.n_pages_with_blocks} pages; the crawl cannot bound the "
        f"byte estimate"
    )

    ratio = predicted / measured.bytes_saved_raw
    assert (top500.BOUNDS.bytes_page_ratio_min <= ratio
            <= top500.BOUNDS.bytes_page_ratio_max_raw), (
        f"predicted {predicted/1e6:.1f} MB for {len(all_blocked):,} blocked "
        f"requests is {ratio:.2f}x the {measured.bytes_saved_raw/1e6:.1f} MB "
        f"measured raw across {measured.n_pages_with_blocks} pages "
        f"({measured.bytes_saved/1e6:.1f} MB drift-corrected), outside "
        f"[{top500.BOUNDS.bytes_page_ratio_min}, "
        f"{top500.BOUNDS.bytes_page_ratio_max_raw}]"
    )

    # The same delta over the pages whose loads repeat, which is the sharper
    # form of the assertion above and the only page-level statement here that
    # needs the crawl's per-pass tables rather than their pooled sum. The
    # numerator is restricted to the same pages, so this is the same ratio on
    # a subset and not a different quantity.
    #
    # Skipped outright on a crawl that was not repeated, rather than graded
    # with an empty filter. Without the per-pass tables every page counts as
    # stable, the ratio is the unfiltered 1.48, and the band below -- which is
    # sized for a denominator the filter has cleaned -- would fail it. The
    # honest reading of a single pass is that it cannot tell which pages its
    # delta can measure, not that all of them can.
    if not top500.page_delta_instability():
        pytest.skip("per-pass paired tables absent; `page_delta_instability` "
                    "needs data/raw/firefox_crawl_500_tracking_x10/paired/"
                    "rep*/paired_pages.csv to say which pages repeat")

    unstable = top500.unstable_pages()
    stable_blocked = [r for r in all_blocked if r.page_idx not in unstable]
    stable_sizes, _cpu = top500.predict(stable_blocked, include_followups=True)
    stable_predicted = sum(stable_sizes)
    assert measured.bytes_saved_raw_stable > 0, (
        f"the whole-page delta over the "
        f"{measured.n_pages_with_blocks - measured.n_pages_unstable} stable "
        f"pages is not positive "
        f"({measured.bytes_saved_raw_stable/1e6:.1f} MB); either the crawl is "
        f"a single pass -- `page_delta_instability` needs "
        f"data/raw/firefox_crawl_500_tracking_x10/paired/rep*/ -- or the "
        f"filter has taken the instrument apart"
    )

    stable = stable_predicted / measured.bytes_saved_raw_stable
    assert (top500.BOUNDS.bytes_page_ratio_min_stable <= stable
            <= top500.BOUNDS.bytes_page_ratio_max_stable), (
        f"predicted {stable_predicted/1e6:.1f} MB is {stable:.2f}x the "
        f"{measured.bytes_saved_raw_stable/1e6:.1f} MB the two arms differed "
        f"by over the "
        f"{measured.n_pages_with_blocks - measured.n_pages_unstable} pages "
        f"whose loads repeat, outside "
        f"[{top500.BOUNDS.bytes_page_ratio_min_stable}, "
        f"{top500.BOUNDS.bytes_page_ratio_max_stable}] -- over all "
        f"{measured.n_pages_with_blocks} pages the same ratio is "
        f"{ratio:.2f}x, so check `page_delta_instability` and the "
        f"{measured.n_pages_unstable} pages it dropped before the estimator"
    )

    assert measured.bytes_saved_tracking_raw > 0, (
        f"the tracker-only byte delta is not positive "
        f"({measured.bytes_saved_tracking_raw/1e6:.1f} MB); either the crawl "
        f"tables predate `bytes_saved_tracking` -- regenerate them with "
        f"src/compare_tracking_arms.py -- or the two arms blocked alike"
    )
    tracking = predicted / measured.bytes_saved_tracking_raw
    assert (top500.BOUNDS.bytes_tracking_ratio_min <= tracking
            <= top500.BOUNDS.bytes_tracking_ratio_max), (
        f"predicted {predicted/1e6:.1f} MB is {tracking:.2f}x the "
        f"{measured.bytes_saved_tracking_raw/1e6:.1f} MB the two arms differed "
        f"by over Disconnect-matched requests, outside "
        f"[{top500.BOUNDS.bytes_tracking_ratio_min}, "
        f"{top500.BOUNDS.bytes_tracking_ratio_max}] -- this is the sharpest "
        f"page-level bound in the module, so read it before the wider ones"
    )


def test_tracking_ratio_holds_per_category() -> None:
    """The sharp bound again, on one kind of page at a time.

    The whole-crawl ratio is one number over 271 pages and a composition error
    cancels in it: 30% high on news against the same number of bytes low on
    productivity leaves it at 1.04 and passes, which
    `test_a_composition_error_is_detectable_only_per_category` asserts rather
    than assumes. The per-request bounds have been cut by category since the
    beginning for that reason; the page-level ones were not, because the
    whole-page delta has no power at that grain -- its churn standard error
    runs into the hundreds of percent per category, which is the crawl
    talking rather than the estimator.

    The Disconnect-only delta does have power there, and which categories it
    has power on is decided by measurement rather than by hand:
    `top500.tracking_churn_se_pct` puts news at 5.6% and productivity at 8.3%
    against 22% to 49% for the five smaller categories, and
    `gradeable_tracking_se_pct` cuts between them. Those two carry 426 MB of
    the 583 MB predicted, so this is not a bound on a corner of the crawl.

    Measured: news 0.95x, productivity 1.03x, against a whole-crawl 1.04x --
    recorded per category in `top500.CATEGORY_TRACKING_RATIO`. They sit much
    closer together than the single-pass crawl's 1.26 and 1.61 did, which is
    what recalibrating the cascade against this denominator does rather than
    evidence that the categories agree.
    """
    pages = [p for p in top500.load_pages() if p.n_blocked_tracking > 0]
    totals = top500.followup_totals_by_page()
    gradeable, se_by_category = {}, {}
    for category in top500.CATEGORIES:
        group = [p for p in pages if p.category == category]
        if not group:
            continue
        se_by_category[category] = se = top500.tracking_churn_se_pct(group)
        if se <= top500.BOUNDS.gradeable_tracking_se_pct:
            gradeable[category] = group

    assert gradeable, (
        "no category's Disconnect-only delta is quiet enough to bound "
        f"(churn standard errors "
        f"{', '.join(f'{k} {v:.0f}%' for k, v in se_by_category.items())}) -- "
        f"the crawl has stopped being able to say anything at this grain"
    )
    assert set(gradeable) == set(top500.CATEGORY_TRACKING_RATIO), (
        f"the categories quiet enough to grade are {sorted(gradeable)} but the "
        f"recorded ratios cover {sorted(top500.CATEGORY_TRACKING_RATIO)} "
        f"(churn standard errors "
        f"{', '.join(f'{k} {v:.0f}%' for k, v in se_by_category.items())}) -- "
        f"measure the newcomer and record it in CATEGORY_TRACKING_RATIO, or "
        f"drop the one that went quiet; the set is part of the assertion"
    )
    covered = sum(sum(totals.get(p.idx, 0) for p in group)
                  for group in gradeable.values())
    assert covered >= 0.5 * sum(totals.get(p.idx, 0) for p in pages), (
        f"the gradeable categories {sorted(gradeable)} carry only "
        f"{covered/1e6:.1f} MB of the {sum(totals.values())/1e6:.1f} MB "
        f"predicted, too little of the crawl for this to stand in for the "
        f"category split"
    )

    aggregate = (sum(totals.get(p.idx, 0) for p in pages)
                 / sum(p.bytes_saved_tracking for p in pages))
    tolerance = top500.BOUNDS.category_tracking_ratio_tolerance_pct
    for category, group in sorted(gradeable.items()):
        predicted = sum(totals.get(p.idx, 0) for p in group)
        delta = sum(p.bytes_saved_tracking for p in group)
        ratio = predicted / delta
        recorded = top500.CATEGORY_TRACKING_RATIO[category]
        off_pct = 100.0 * (ratio - recorded) / recorded
        assert abs(off_pct) <= tolerance, (
            f"{category}: predicted {predicted/1e6:.1f} MB over "
            f"{len(group)} pages is {ratio:.2f}x the {delta/1e6:.1f} MB the "
            f"two arms differed by over Disconnect-matched requests there, "
            f"{off_pct:+.0f}% from the {recorded:.2f}x recorded for it and "
            f"outside the {tolerance:.0f}% tolerance -- a category-sized "
            f"error the whole-crawl ratio of {aggregate:.2f}x cannot see "
            f"(churn standard error here {se_by_category[category]:.1f}%)"
        )


def test_tracking_ratio_holds_on_every_random_half_crawl() -> None:
    """And on every random half, not merely on the median of them.

    `test_page_level_ratio_holds_on_random_half_crawls` bounds the median and
    nothing else, because its denominator is a difference of two noisy
    whole-page totals: on an unlucky half it comes out near zero and the ratio
    explodes, to 10.0 at the 99th percentile and 13.6 at worst. A per-subset
    bound there would be a bound on the seed.

    The quiet denominator behaves, which is what makes the stronger statement
    affordable: over the same 200 seeded halves the ratio stays inside
    [0.78, 1.42] with a median of 1.05, so every half can be asserted rather
    than the middle of them. The median is checked too, against the
    whole-crawl band -- halving the crawl should widen the spread and leave
    the centre alone.

    This is the family of bounds ten passes bought nothing for, and the width
    above is the reason to keep them anyway: a half-crawl is 136 of the same
    271 pages, so what [0.78, 1.42] measures is page-set variation, which
    repeating the crawl cannot touch.
    """
    totals = top500.followup_totals_by_page()
    ratios = []
    for subset in top500.random_page_subsets():
        delta = sum(p.bytes_saved_tracking for p in subset)
        if delta <= 0:
            continue
        ratios.append(sum(totals.get(p.idx, 0) for p in subset) / delta)

    assert len(ratios) >= 100, f"only {len(ratios)} usable subsets"
    low, high = min(ratios), max(ratios)
    assert (top500.BOUNDS.subset_bytes_tracking_ratio_min <= low
            and high <= top500.BOUNDS.subset_bytes_tracking_ratio_max), (
        f"over {len(ratios)} random half-crawls the ratio to the "
        f"Disconnect-matched delta runs [{low:.2f}, {high:.2f}], outside "
        f"[{top500.BOUNDS.subset_bytes_tracking_ratio_min}, "
        f"{top500.BOUNDS.subset_bytes_tracking_ratio_max}] -- some half of "
        f"the pages carries a page-level error the whole crawl hides"
    )
    median = statistics.median(ratios)
    assert (top500.BOUNDS.bytes_tracking_ratio_min <= median
            <= top500.BOUNDS.bytes_tracking_ratio_max), (
        f"the median random half-crawl predicts {median:.2f}x, outside the "
        f"whole-crawl band [{top500.BOUNDS.bytes_tracking_ratio_min}, "
        f"{top500.BOUNDS.bytes_tracking_ratio_max}] -- halving the crawl "
        f"should move the spread, not the centre"
    )


def test_no_single_page_carries_the_tracking_ratio() -> None:
    """Drop any one page and the sharp ratio barely moves.

    The twin of `test_no_single_page_carries_the_page_level_ratio`, on the
    quiet denominator and so with a bound worth two and a half times as much:
    that one has to tolerate 19.6%, because a single page carries that much of
    the whole-page saving, where here no page carries more than 8.2%.

    Measured: +8.2% at worst and -1.3% the other way. The worst is
    optimizely.com, which sheds 45.5 MB of listed bytes over ten passes --
    8.1% of the denominator -- against 60 blocked requests and 3.5 MB of
    estimate, so dropping it moves the denominator and barely touches the
    numerator.
    """
    totals = top500.followup_totals_by_page()
    pages = [p for p in top500.load_pages() if p.n_blocked_tracking > 0]
    full = (sum(totals.get(p.idx, 0) for p in pages)
            / sum(p.bytes_saved_tracking for p in pages))

    worst_page, worst_shift = None, 0.0
    for dropped in pages:
        rest = [p for p in pages if p.idx != dropped.idx]
        delta = sum(p.bytes_saved_tracking for p in rest)
        if delta <= 0:
            continue
        ratio = sum(totals.get(p.idx, 0) for p in rest) / delta
        shift = 100.0 * (ratio - full) / full
        if abs(shift) > abs(worst_shift):
            worst_page, worst_shift = dropped, shift

    assert abs(worst_shift) <= \
        top500.BOUNDS.max_single_page_tracking_ratio_shift_pct, (
            f"dropping {worst_page.url if worst_page else '?'} alone moves the "
            f"Disconnect-matched ratio by {worst_shift:+.1f}% (from "
            f"{full:.2f}x), outside the "
            f"{top500.BOUNDS.max_single_page_tracking_ratio_shift_pct:.0f}% "
            f"bound -- one page is carrying the sharpest bound in the module"
        )


def test_blocked_bytes_are_within_bounds_against_http_archive() -> None:
    """The same total, against a crawl this repo did not run.

    Every other bound in this module divides by something the paired crawl
    measured on one afternoon. A systematic fault in that crawl -- a
    misconfigured arm, a bad network hour, a Playwright upgrade changing what
    loads -- would move the estimator's grade and every bound with it, and
    nothing here would notice. This one divides by HTTP Archive's own crawl
    of the same 500 domains, which is a different browser, a different month
    and a different set of machines.

    It is loose on purpose: 0.20x measured, bounded to [0.14, 0.28]. The two
    quantities differ by a roughly constant factor for reasons that are not
    about the estimator being right or wrong -- HTTP Archive's pages are
    heavier than the crawl's mobile ones, its tracker mass includes entries
    ETP would not block, and it cannot see what a blocked tracker would have
    pulled in from a host the Disconnect list does not name. What survives
    all that is a scale check: the estimator's total has to stay the right
    size against a population it has never been near. See `HTTP_ARCHIVE_CSV`.

    Skipped when `data/tranco_500_http_archive.csv` is absent; rebuild it with
    `src/build_tranco_500_http_archive.py`, which needs the 50% export.
    """
    if not top500.load_http_archive_mass():
        pytest.skip(f"{top500.HTTP_ARCHIVE_CSV.name} is absent; rebuild it "
                    f"with src/build_tranco_500_http_archive.py")

    predicted_by_page = top500.followup_totals_by_page()
    covered = [(p, ha) for p in top500.load_pages()
               if p.n_blocked_tracking > 0
               and (ha := top500.http_archive_for(p)) is not None
               and ha.tracker_bytes > 0]
    touched = sum(1 for p in top500.load_pages() if p.n_blocked_tracking > 0)
    assert len(covered) >= 0.5 * touched, (
        f"HTTP Archive covers only {len(covered)} of {touched} pages ETP acted "
        f"on, too few to bound the estimate against"
    )

    # The only denominator here that is not ten passes' worth: HTTP Archive
    # crawled each domain once. See `top500.n_passes`.
    predicted = (sum(predicted_by_page.get(p.idx, 0) for p, _ha in covered)
                 / top500.n_passes())
    mass = sum(ha.tracker_bytes for _p, ha in covered)
    ratio = predicted / mass
    assert (top500.BOUNDS.external_ratio_min <= ratio
            <= top500.BOUNDS.external_ratio_max), (
        f"predicted {predicted/1e6:.1f} MB on the {len(covered)} pages HTTP "
        f"Archive also crawled is {ratio:.2f}x the {mass/1e6:.1f} MB of "
        f"tracker traffic it measured there, outside "
        f"[{top500.BOUNDS.external_ratio_min}, "
        f"{top500.BOUNDS.external_ratio_max}] -- the one bound here that does "
        f"not depend on this repo's own crawl being sound"
    )


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #
# Everything above is one number over 271 pages, and a number like that can be
# carried by a handful of them. These cut the crawl up -- at random, and by
# category above -- and ask whether the same statements survive. Random
# subsets are seeded (`top500.random_page_subsets`), so a failure is a real
# change and not a new draw.


def test_byte_bias_holds_on_random_half_crawls() -> None:
    """Halve the crawl at random, 200 times, and the bias stays put.

    The aggregate bias is +2.1% over 11,290 priced requests. That could be two
    dozen pages cancelling a systematic error on the rest, and the category
    split would not necessarily see it -- `productivity` alone is 122 pages.
    This asks the question the category split cannot: is the estimate
    unbiased on an *arbitrary* half of the crawl, not just on the halves
    someone thought to name?

    Measured over the 200 seeded halves: median +1.5%, close to the whole
    crawl's +2.1%, with the 1st and 99th percentiles at -8.6% and +14.5% and
    the worst single half at 18.4%. The per-subset bar is looser than
    `max_abs_bytes_bias_pct` because half a crawl is a noisier instrument;
    the bar on the median is tighter, because halving should widen the spread
    and leave the centre alone.
    """
    totals = top500.matched_totals_by_page()
    biases = []
    for subset in top500.random_page_subsets():
        pred = sum(totals.get(p.idx, (0, 0))[0] for p in subset)
        obs = sum(totals.get(p.idx, (0, 0))[1] for p in subset)
        if obs <= 0:
            continue
        biases.append(100.0 * (pred - obs) / obs)

    assert len(biases) >= 100, f"only {len(biases)} usable subsets"
    worst = max(biases, key=abs)
    assert abs(worst) <= top500.BOUNDS.subset_max_abs_bytes_bias_pct, (
        f"the worst of {len(biases)} random half-crawls is {worst:+.1f}% off, "
        f"outside the {top500.BOUNDS.subset_max_abs_bytes_bias_pct:.0f}% bound "
        f"-- some subset of pages is carrying a bias the whole-crawl total hides"
    )
    median = statistics.median(biases)
    assert abs(median) <= top500.BOUNDS.subset_median_abs_bytes_bias_pct, (
        f"the median random half-crawl is {median:+.1f}% off, outside the "
        f"{top500.BOUNDS.subset_median_abs_bytes_bias_pct:.0f}% bound"
    )


def test_byte_bias_degrades_gracefully_on_quarter_crawls() -> None:
    """A quarter of the crawl is noisier, and must be noisier in the right way.

    The point is not the bound, which is loose. It is that the *median* of the
    quarters stays on the whole-crawl figure while the spread widens: that is
    what a sampling error looks like. A bias that grew as the subsets shrank
    would mean the estimate depends on which pages are in it, which is the
    failure mode a single aggregate cannot show.

    Measured: median +1.9%, p1/p99 -17.4% and +23.6%, worst 34.6%.
    """
    totals = top500.matched_totals_by_page()
    halves, quarters = [], []
    for fraction, out in ((0.5, halves), (0.25, quarters)):
        for subset in top500.random_page_subsets(fraction=fraction):
            pred = sum(totals.get(p.idx, (0, 0))[0] for p in subset)
            obs = sum(totals.get(p.idx, (0, 0))[1] for p in subset)
            if obs > 0:
                out.append(100.0 * (pred - obs) / obs)

    worst = max(quarters, key=abs)
    assert abs(worst) <= top500.BOUNDS.quarter_subset_max_abs_bytes_bias_pct, (
        f"the worst random quarter-crawl is {worst:+.1f}% off, outside the "
        f"{top500.BOUNDS.quarter_subset_max_abs_bytes_bias_pct:.0f}% bound"
    )
    assert abs(statistics.median(quarters)) <= \
        top500.BOUNDS.subset_median_abs_bytes_bias_pct, (
            f"the median random quarter-crawl is "
            f"{statistics.median(quarters):+.1f}% off, which is a bias rather "
            f"than sampling noise -- the median should not move with the "
            f"subset size"
        )
    spread = statistics.pstdev(quarters) / statistics.pstdev(halves)
    assert spread > 1.0, (
        f"quarter-crawls are no more variable than half-crawls "
        f"(sd ratio {spread:.2f}), so the subsets are not independent draws "
        f"and these tests are measuring nothing"
    )


def test_page_level_ratio_holds_on_random_half_crawls(
        measured: top500.MeasuredPageCost) -> None:
    """The cascade-inclusive ratio survives halving the crawl.

    Only the median is bounded tightly, and the reason is in the denominator:
    a random half's measured saving is a difference of two noisy page totals,
    so on an unlucky half it comes out near zero and the ratio explodes. The
    99th percentile across halves is 10.0 against a median of 1.28. Bounding
    the tail tightly would be bounding the crawl's noise, so the tail gets the
    wide band instead and the median carries the statement.
    """
    totals = top500.followup_totals_by_page()
    ratios = []
    for subset in top500.random_page_subsets():
        m = top500.measure(subset)
        if m.bytes_saved <= 0:
            continue
        ratios.append(sum(totals.get(p.idx, 0) for p in subset) / m.bytes_saved)

    assert len(ratios) >= 100, f"only {len(ratios)} usable subsets"
    median = statistics.median(ratios)
    assert (top500.BOUNDS.subset_median_bytes_page_ratio_min <= median
            <= top500.BOUNDS.subset_median_bytes_page_ratio_max), (
        f"the median of {len(ratios)} random half-crawls predicts {median:.2f}x "
        f"the measured saving, outside "
        f"[{top500.BOUNDS.subset_median_bytes_page_ratio_min}, "
        f"{top500.BOUNDS.subset_median_bytes_page_ratio_max}] -- the "
        f"whole-crawl ratio is {sum(totals.values())/measured.bytes_saved:.2f}x"
    )
    inside = sum(1 for r in ratios
                 if top500.BOUNDS.wide_band_min <= r <= top500.BOUNDS.wide_band_max)
    share = inside / len(ratios)
    assert share >= top500.BOUNDS.subset_ratio_in_wide_band, (
        f"only {share:.0%} of random half-crawls land in "
        f"[{top500.BOUNDS.wide_band_min}, {top500.BOUNDS.wide_band_max}] "
        f"(needs {top500.BOUNDS.subset_ratio_in_wide_band:.0%})"
    )


def test_no_single_page_carries_the_page_level_ratio(
        measured: top500.MeasuredPageCost) -> None:
    """Drop any one page and the whole-crawl ratio barely moves.

    The complement of the random-subset tests: those ask whether the result
    survives losing half the crawl, this asks whether it depends on keeping
    one particular page. A leave-one-out swing near the bound would mean the
    583 MB of estimate and the 393 MB of measurement are being reconciled by a
    single outlier -- which is exactly how a page-level measurement this noisy
    fails, and on this crawl it half does.

    Measured: -19.6%, on tradingview.com, whose blocking arm transferred
    106.0 MB *more* than its control arm across ten passes. That is -27% of
    the whole-page denominator from one page, and dropping it raises the
    denominator by more than a third. The bound admits it rather than
    excluding the page; the quiet denominator's twin of this test, where the
    worst page moves the ratio by 8.2%, is the one to read instead.
    """
    totals = top500.followup_totals_by_page()
    pages = [p for p in top500.load_pages() if p.n_blocked_tracking > 0]
    full = sum(totals.values()) / measured.bytes_saved

    worst_page, worst_shift = None, 0.0
    for dropped in pages:
        rest = [p for p in pages if p.idx != dropped.idx]
        m = top500.measure(rest)
        if m.bytes_saved <= 0:
            continue
        ratio = sum(totals.get(p.idx, 0) for p in rest) / m.bytes_saved
        shift = 100.0 * (ratio - full) / full
        if abs(shift) > abs(worst_shift):
            worst_page, worst_shift = dropped, shift

    assert abs(worst_shift) <= top500.BOUNDS.max_single_page_ratio_shift_pct, (
        f"dropping {worst_page.url if worst_page else '?'} alone moves the "
        f"page-level ratio by {worst_shift:+.1f}% (from {full:.2f}x), outside "
        f"the {top500.BOUNDS.max_single_page_ratio_shift_pct:.0f}% bound -- one "
        f"page is carrying the comparison"
    )


def test_cpu_estimate_is_within_bounds(
        all_blocked: list[top500.BlockedRequest],
        predicted_cpu_s: float,
        measured: top500.MeasuredPageCost) -> None:
    """Estimated CPU for blocked trackers sits inside a plausible band.

    Compared against CPU measured at page level, drift-corrected, since the
    crawl has no per-request CPU. The ceiling encodes a structural fact rather
    than a fitted one: see `Bounds.cpu_ratio_max`.
    """
    predicted_s = predicted_cpu_s
    assert measured.cpu_saved_s > 0, (
        f"measured CPU saving is not positive ({measured.cpu_saved_s:.1f} s); "
        f"the crawl cannot bound the CPU estimate"
    )

    ratio = predicted_s / measured.cpu_saved_s
    assert top500.BOUNDS.cpu_ratio_min <= ratio <= top500.BOUNDS.cpu_ratio_max, (
        f"predicted CPU {predicted_s:.1f} s for {len(all_blocked):,} blocked "
        f"requests is {ratio:.2f}x the {measured.cpu_saved_s:.1f} s measured "
        f"across {measured.n_pages_with_blocks} pages, outside "
        f"[{top500.BOUNDS.cpu_ratio_min}, {top500.BOUNDS.cpu_ratio_max}]"
    )


def test_cpu_estimate_does_not_exceed_measured_saving(
        predicted_cpu_s: float,
        measured: top500.MeasuredPageCost) -> None:
    """A per-request CPU estimate must stay under the page-level saving.

    This restates `Bounds.cpu_ratio_max` on purpose, so that widening the band
    for some future measurement cannot quietly drop the invariant with it.

    Blocking a tracker also prevents the subresources it would have fetched and
    run, so the measured page-level CPU saving includes work the estimator
    never claims to account for. Predicting *more* than was measured would mean
    it is over-attributing -- the same inequality the byte estimate has to
    satisfy, and the one that caught an over-count of 4.4x when the population
    was chosen by a bare `is_tracker()` lookup.
    """
    predicted_s = predicted_cpu_s
    assert predicted_s <= measured.cpu_saved_s, (
        f"predicted CPU {predicted_s:.1f} s exceeds the {measured.cpu_saved_s:.1f} s "
        f"measured, so the estimate is claiming cycles the crawl did not see saved"
    )


def test_zero_byte_beacons_still_cost_cpu(
        blocked: list[top500.BlockedRequest]) -> None:
    """Requests that transfer nothing are not free.

    11% of the blocked requests carry no bytes, and something still dispatched
    them and ran their completion handler. Without the fixed per-request term
    in `cpu_ms_for` every one of them would be estimated at zero CPU, which
    would quietly zero out a tenth of the population.
    """
    zero_byte = [r for r in blocked if r.observed_bytes == 0]
    if not zero_byte:
        pytest.skip("no zero-byte blocked requests in this crawl")

    _sizes, cpu_ms = top500.predict(zero_byte)
    assert all(ms > 0 for ms in cpu_ms), (
        f"{sum(1 for ms in cpu_ms if ms <= 0)} of {len(zero_byte)} zero-byte "
        f"requests were estimated at no CPU cost"
    )


def test_cpu_rate_is_ordered_by_context() -> None:
    """A script costs more CPU per byte than a font or a video does.

    `CPU_MS_PER_KIB` is indexed by the `RequestContext` discriminant, so
    renumbering the enum -- which the table's keys also depend on -- would
    silently pair every context with the wrong rate. Nothing else would catch
    that: the byte estimate would still look right.
    """
    import llm_classifier
    from llm_classifier import RequestContext as RC

    url = "https://tracker.example/asset"
    same_size = {}
    for name in ("SCRIPT", "CSS", "IMAGE", "FONT", "VIDEO"):
        ctx = getattr(RC, name)
        size, cpu = llm_classifier.estimate_resources(
            url, ctx, llm_classifier.RequestInitiator.UNKNOWN, "GET")
        same_size[name] = (size, cpu)

    # Compare rates, not totals: each context resolves its own byte estimate.
    def rate(name: str) -> float:
        size, cpu = same_size[name]
        return cpu / max(size / 1024.0, 1e-9)

    assert rate("SCRIPT") > rate("CSS") > rate("FONT"), (
        "CPU per KiB should fall from scripts through stylesheets to fonts, got "
        + ", ".join(f"{n}={rate(n):.3f}" for n in ("SCRIPT", "CSS", "FONT"))
    )
    assert rate("SCRIPT") > rate("VIDEO"), (
        f"scripts should cost more main-thread time per KiB than video "
        f"({rate('SCRIPT'):.3f} vs {rate('VIDEO'):.3f})"
    )


@pytest.mark.parametrize("name", sorted(top500.BROKEN_SCALES))
def test_broken_estimates_are_detectably_bad(
        name: str, blocked: list[top500.BlockedRequest],
        all_blocked: list[top500.BlockedRequest],
        predicted: tuple[list[int], list[float]],
        predicted_cpu_s: float,
        measured: top500.MeasuredPageCost) -> None:
    """Negative control: a mis-scaled estimator must fail these bounds.

    If scaling every prediction by 10x, 0.1x or 0 still passed, the bounds
    above would not be measuring prediction quality and the whole module would
    be decorative.
    """
    scale = top500.BROKEN_SCALES[name]
    sizes, _cpu_ms = predicted
    observed = sum(r.observed_bytes for r in blocked)
    # With the follow-ups, as the page-level bound itself is scored.
    all_sizes, _ = top500.predict(all_blocked, include_followups=True)

    bias_pct = 100.0 * (sum(sizes) * scale - observed) / observed
    cpu_ratio = predicted_cpu_s * scale / measured.cpu_saved_s
    bytes_page_ratio = sum(all_sizes) * scale / measured.bytes_saved

    tracking_ratio = (sum(all_sizes) * scale
                      / measured.bytes_saved_tracking_raw)
    # The stable whole-page bound is in here too, over the same pages it is
    # asserted on, so the control covers every band
    # `test_blocked_bytes_are_within_bounds_against_the_raw_page_delta`
    # enforces rather than the two it used to.
    unstable = top500.unstable_pages()
    stable_sizes, _ = top500.predict(
        [r for r in all_blocked if r.page_idx not in unstable],
        include_followups=True)
    stable_ratio = (sum(stable_sizes) * scale
                    / measured.bytes_saved_raw_stable)

    bytes_ok = abs(bias_pct) <= top500.BOUNDS.max_abs_bytes_bias_pct
    cpu_ok = (top500.BOUNDS.cpu_ratio_min <= cpu_ratio
              <= top500.BOUNDS.cpu_ratio_max)
    bytes_page_ok = (top500.BOUNDS.bytes_page_ratio_min <= bytes_page_ratio
                     <= top500.BOUNDS.bytes_page_ratio_max)
    stable_ok = (top500.BOUNDS.bytes_page_ratio_min_stable <= stable_ratio
                 <= top500.BOUNDS.bytes_page_ratio_max_stable)
    tracking_ok = (top500.BOUNDS.bytes_tracking_ratio_min <= tracking_ratio
                   <= top500.BOUNDS.bytes_tracking_ratio_max)
    assert not (bytes_ok and cpu_ok and bytes_page_ok and stable_ok
                and tracking_ok), (
        f"scaling every estimate by {scale}x still passed every bound "
        f"(bytes {bias_pct:+.1f}%, CPU {cpu_ratio:.2f}x, "
        f"page-level bytes {bytes_page_ratio:.2f}x, stable page-level "
        f"{stable_ratio:.2f}x, tracker-only "
        f"{tracking_ratio:.2f}x) -- the bounds are not sensitive to "
        f"prediction quality"
    )


def test_a_composition_error_is_detectable_only_per_category() -> None:
    """Negative control for the category bound, and for why it is needed.

    Every predictor in `BROKEN_SCALES` is wrong by a factor everywhere, so
    the whole-crawl bounds catch it and nothing is learned about whether the
    finer ones pull their weight. `top500.COMPOSITION_SKEW` is the error they
    cannot see: 30% too high on news and the same number of bytes too low on
    productivity, which leaves the crawl-wide ratio at 1.04 -- inside
    [0.94, 1.15] -- while both categories land more than 20% from the ratios
    recorded for them in `top500.CATEGORY_TRACKING_RATIO`.

    So this asserts both halves. The aggregate must pass, because a
    composition error is exactly what a total hides, and if it started
    failing this control would have stopped being about composition. The
    category bound must fail.
    """
    category_a, category_b, skew = top500.COMPOSITION_SKEW
    pages = [p for p in top500.load_pages() if p.n_blocked_tracking > 0]
    totals = top500.followup_totals_by_page()
    in_a = sum(totals.get(p.idx, 0) for p in pages if p.category == category_a)
    in_b = sum(totals.get(p.idx, 0) for p in pages if p.category == category_b)
    assert in_a > 0 and in_b > 0, (
        f"{category_a} or {category_b} has no predicted bytes, so the skew "
        f"cannot be applied -- has the category file changed?"
    )
    # Lift one category and drop the other by the same number of bytes, each
    # spread over its pages in proportion to what was predicted there.
    moved = skew * in_a
    factor = {category_a: 1.0 + skew, category_b: 1.0 - moved / in_b}
    skewed = {p.idx: totals.get(p.idx, 0) * factor.get(p.category, 1.0)
              for p in pages}

    def predicted(group: list[top500.CrawledPage]) -> float:
        return sum(skewed[p.idx] for p in group)

    aggregate = predicted(pages) / sum(p.bytes_saved_tracking for p in pages)
    assert (top500.BOUNDS.bytes_tracking_ratio_min <= aggregate
            <= top500.BOUNDS.bytes_tracking_ratio_max), (
        f"moving {moved/1e6:.1f} MB from {category_b} to {category_a} changed "
        f"the whole-crawl ratio to {aggregate:.2f}x, so this control is no "
        f"longer testing what a total cannot see -- pick a skew that cancels"
    )

    outside = []
    for category in (category_a, category_b):
        group = [p for p in pages if p.category == category]
        ratio = predicted(group) / sum(p.bytes_saved_tracking for p in group)
        recorded = top500.CATEGORY_TRACKING_RATIO[category]
        if abs(100.0 * (ratio - recorded) / recorded) > \
                top500.BOUNDS.category_tracking_ratio_tolerance_pct:
            outside.append(f"{category} {ratio:.2f}x against {recorded:.2f}x")

    assert outside, (
        f"a {skew:.0%} composition error between {category_a} and "
        f"{category_b} stayed inside the "
        f"{top500.BOUNDS.category_tracking_ratio_tolerance_pct:.0f}% "
        f"per-category tolerance while the aggregate sat at "
        f"{aggregate:.2f}x -- the category bound is not catching anything the "
        f"whole-crawl one misses, which was its whole purpose"
    )


@pytest.mark.parametrize("name", sorted(top500.BROKEN_SCALES))
def test_broken_estimates_are_detectable_on_subsets_too(name: str) -> None:
    """The subset bounds are a real net, not a wider hole.

    The subset tests are looser than the whole-crawl ones by construction --
    a half-crawl is a noisier instrument, and the page-level ratio's tail is
    bounded at [0.4, 2.5] rather than [0.75, 1.25]. Loose bounds that nothing
    can fail would be worse than no bounds, because they would read as
    coverage. So every mis-scaled estimator has to fail here as well, and
    what the bounds tolerate is recorded by the fact that 0.1x and 10x do not
    survive them.
    """
    scale = top500.BROKEN_SCALES[name]
    matched = top500.matched_totals_by_page()
    followups = top500.followup_totals_by_page()

    biases, ratios, tracking = [], [], []
    for subset in top500.random_page_subsets():
        pred = sum(matched.get(p.idx, (0, 0))[0] for p in subset) * scale
        obs = sum(matched.get(p.idx, (0, 0))[1] for p in subset)
        if obs > 0:
            biases.append(100.0 * (pred - obs) / obs)
        m = top500.measure(subset)
        if m.bytes_saved > 0:
            ratios.append(sum(followups.get(p.idx, 0) for p in subset)
                          * scale / m.bytes_saved)
        if m.bytes_saved_tracking_raw > 0:
            tracking.append(sum(followups.get(p.idx, 0) for p in subset)
                            * scale / m.bytes_saved_tracking_raw)

    worst_bias = max(biases, key=abs)
    median_ratio = statistics.median(ratios)
    inside = sum(1 for r in ratios
                 if top500.BOUNDS.wide_band_min <= r <= top500.BOUNDS.wide_band_max)

    bias_ok = abs(worst_bias) <= top500.BOUNDS.subset_max_abs_bytes_bias_pct
    median_ok = (top500.BOUNDS.subset_median_bytes_page_ratio_min <= median_ratio
                 <= top500.BOUNDS.subset_median_bytes_page_ratio_max)
    band_ok = inside / len(ratios) >= top500.BOUNDS.subset_ratio_in_wide_band
    tracking_ok = (top500.BOUNDS.subset_bytes_tracking_ratio_min <= min(tracking)
                   and max(tracking)
                   <= top500.BOUNDS.subset_bytes_tracking_ratio_max)
    assert not (bias_ok and median_ok and band_ok and tracking_ok), (
        f"scaling every estimate by {scale}x still passed every subset bound "
        f"(worst half-crawl bias {worst_bias:+.1f}%, median ratio "
        f"{median_ratio:.2f}x, {inside/len(ratios):.0%} in the wide band, "
        f"Disconnect-matched ratio over halves "
        f"[{min(tracking):.2f}, {max(tracking):.2f}])"
    )
