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


@pytest.mark.parametrize("stratum", top500.STRATA)
def test_byte_bias_is_within_bounds_per_stratum(stratum: str) -> None:
    """Each kind of page is unbiased on its own, not just the crawl as a whole.

    `test_blocked_byte_total_is_within_bounds` sums over all 500 domains, so a
    bias confined to one kind of page cancels against the others and passes.
    It did: news and media pages ran ~19% low while the rest of the crawl ran
    +0.1%, and the total landed at -7.1%, comfortably inside the bound. That is
    the case the dashboard most needs to be right about, since ad-heavy
    editorial pages are where blocking saves the most. The host rungs in
    `llm-classifier` closed it to -8.5% against `other` at +1.1%; this test is
    what would catch it reopening.

    The bar is the same on every stratum on purpose -- there is no reason a
    news page should be allowed more error than a bank's. A stratum that misses
    it must be recorded in `KNOWN_BIASED_STRATA` with its measured bias, and
    only the ones in `BIASED_STRATA_SKIPPED` are let off with a skip; any other
    miss is a hard failure. So a stratum that goes bad later fails loudly
    rather than passing a threshold widened to fit it.
    """
    requests = top500.load_blocked_by_stratum()[stratum]
    assert requests, f"stratum {stratum!r} has no priced blocked requests"
    observed = sum(r.observed_bytes for r in requests)
    assert observed > 0, f"stratum {stratum!r} has no observed bytes"

    sizes, _cpu = top500.predict(requests)
    bias_pct = 100.0 * (sum(sizes) - observed) / observed

    known = top500.KNOWN_BIASED_STRATA.get(stratum)
    if known is not None and abs(bias_pct) > top500.BOUNDS.max_abs_bytes_bias_pct:
        if stratum in top500.BIASED_STRATA_SKIPPED:
            pytest.skip(f"{stratum} is known to be biased: {known} "
                        f"(measured {bias_pct:+.1f}%), skipping")
        pytest.fail(f"{known} (measured {bias_pct:+.1f}%)")

    assert abs(bias_pct) <= top500.BOUNDS.max_abs_bytes_bias_pct, (
        f"{stratum}: blocked bytes estimated {bias_pct:+.1f}% off "
        f"({sum(sizes)/1e6:.1f} MB predicted vs {observed/1e6:.1f} MB observed) "
        f"over {len(requests):,} requests on this stratum alone, outside the "
        f"{top500.BOUNDS.max_abs_bytes_bias_pct:.0f}% bound -- the aggregate "
        f"test can miss this when other strata cancel it"
    )


def test_stratum_bias_is_not_hidden_by_the_aggregate() -> None:
    """The stratified test must be able to see what the aggregate cannot.

    If the strata ever collapsed -- one of them emptied by a loader change, or
    every page landing in `other` because host matching broke -- the test above
    would still pass while measuring nothing. This asserts the split is real
    and that it is actually sharper than the grand total.
    """
    by_stratum = top500.load_blocked_by_stratum()
    biases = {}
    for name, requests in by_stratum.items():
        observed = sum(r.observed_bytes for r in requests)
        assert len(requests) >= 50, (
            f"stratum {name!r} has only {len(requests)} priced requests, too "
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
        f"no stratum is further off than the {aggregate:+.1f}% aggregate "
        f"({', '.join(f'{k} {v:+.1f}%' for k, v in biases.items())}), so "
        f"stratifying is measuring nothing the grand total did not already show"
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
    """Price *every* tracking block, including the 728 with no priced twin.

    `test_blocked_byte_total_is_within_bounds` is the sharper check but it can
    only grade the 61.5% of blocks whose URL recurred across the two arms; the
    rest have no per-request observation to compare against. This one covers
    all 1,890 by comparing against the page-level byte saving, the same way the
    CPU tests do, so a regression confined to the unmatched population cannot
    pass unnoticed.

    It is one-sided in practice: the page-level saving also contains the
    subresources a blocked tracker would have gone on to fetch, which the
    estimator does not model, so the measured figure is several times the
    predicted one and the floor is correspondingly low. See
    `Bounds.bytes_page_ratio_min`.
    """
    sizes, _cpu = top500.predict(all_blocked)
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
    all_sizes, _ = top500.predict(all_blocked)

    bias_pct = 100.0 * (sum(sizes) * scale - observed) / observed
    cpu_ratio = predicted_cpu_s * scale / measured.cpu_saved_s
    bytes_page_ratio = sum(all_sizes) * scale / measured.bytes_saved

    bytes_ok = abs(bias_pct) <= top500.BOUNDS.max_abs_bytes_bias_pct
    cpu_ok = (top500.BOUNDS.cpu_ratio_min <= cpu_ratio
              <= top500.BOUNDS.cpu_ratio_max)
    bytes_page_ok = (top500.BOUNDS.bytes_page_ratio_min <= bytes_page_ratio
                     <= top500.BOUNDS.bytes_page_ratio_max)
    assert not (bytes_ok and cpu_ok and bytes_page_ok), (
        f"scaling every estimate by {scale}x still passed every bound "
        f"(bytes {bias_pct:+.1f}%, CPU {cpu_ratio:.2f}x, "
        f"page-level bytes {bytes_page_ratio:.2f}x) -- the bounds are not "
        f"sensitive to prediction quality"
    )
