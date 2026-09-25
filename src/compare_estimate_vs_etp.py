"""
Score the shipped `llm-classifier` lookup table against the paired Firefox
crawl: what the table *predicts* a blocked tracker would have cost, versus
what that request really cost when Firefox was allowed to fetch it.

The control arm of the crawl fetched everything, so every request there has a
real `_transferSize`. The blocking arm tells us which requests Firefox
actually refused. Together they give something the HTTP Archive training data
cannot: an in-browser ground truth for the exact quantity the table is asked
for at block time, on pages nobody trained on.

The table is fed only what the browser knows before the response exists -- the
URL and the request context -- so nothing here leaks the answer.

FOUR SEPARATE QUESTIONS, KEPT SEPARATE
--------------------------------------
Conflating these is the easy way to get a flattering number, so each is
reported on its own:

  1. size accuracy, Disconnect population
     Over every control-arm request the Disconnect list calls a tracker,
     predicted vs observed bytes. No request matching is involved and no
     observation is missing, so this is the statistically cleanest read on
     the table itself.

  2. size accuracy, exactly what ETP blocked
     The same comparison restricted to requests Firefox really did block and
     that `compare_tracking_arms.py` could match back to a control-arm
     observation. Smaller and subject to match loss, but it is the true
     deployment population.

  3. classification divergence
     Disconnect-flagged-in-control versus ETP-blocked-in-Firefox, by count and
     by bytes. The table is only ever asked about requests something decided
     to block, so disagreement between the offline list and shipped ETP is a
     first-class error source, not a footnote.

  4. end-to-end savings estimate
     Predicted total savings against measured total savings -- the number a
     privacy dashboard would actually print.

  5. estimated bytes of what ETP blocked, against the page-level delta
     What the table predicts for every request Firefox refused -- unmatched
     ones included, since it prices a URL and needs no control-arm twin --
     against (control transfer - blocking transfer) on those same pages. This
     is the only question here whose ground truth covers *all* the blocking
     rather than the matched subset, and the only one measured the way a
     browser would feel it.

WHY THE ESTIMATE SHOULD NOT MATCH THE PAGE-LEVEL DELTA
-----------------------------------------------------
Total page bytes saved exceeds the bytes of the blocked requests themselves,
because a blocked tracker never gets to inject its own subresources. The table
estimates only the request it is handed, so questions 1 and 2 compare it
against the direct bytes of blocked requests. Question 5 sets it against the
page-level delta anyway, because that is the quantity a user experiences, but
the ratio there is expected to land *below* 1 and a ratio above 1 means the
estimator is over-attributing rather than that it is finally accurate.

Two loads of the same page differ even where nothing was blocked -- ads rotate,
content churns -- so question 5 also reports a drift correction: the mean delta
over pages ETP touched nothing on, which is pure churn by construction,
subtracted per page from the pages it did touch.

Usage:
    python src/compare_estimate_vs_etp.py \\
        --normal   data/raw/firefox_crawl_500_tracking/normal \\
        --private  data/raw/firefox_crawl_500_tracking/private \\
        --blocked  data/raw/firefox_crawl_500_tracking/blocked_observed_bytes.csv \\
        --paired   data/raw/firefox_crawl_500_tracking/paired_pages.csv \\
        --out      data/raw/firefox_crawl_500_tracking
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from urllib.parse import urlsplit

import disconnect
import llm_classifier
from llm_classifier import RequestContext as RC

# What `estimate_resources` is told about requests whose initiator and HTTP
# method the crawl did not record, which is all of them: these logs carry the
# URL, the resource type and the transferred size. Both values mean "not
# known", and both leave the estimate exactly what it was before the estimator
# took either -- the table never keys on the initiator, and a method it cannot
# recognise is read as a GET rather than as a bodyless one.
_UNKNOWN_INITIATOR = llm_classifier.RequestInitiator.UNKNOWN
_UNKNOWN_METHOD = ""

# ETP Standard blocks these Disconnect categories as tracking content. Content
# is excluded (that is Strict), and Cryptomining/Fingerprinting are excluded to
# keep this population the same one the ETP-blocked comparison uses, which is
# filtered to protection=="tracking"; the crawl does block those two in its
# blocking arm, but they are a handful of requests and a different protection.
ETP_CATEGORIES = "Advertising,Analytics,Social"

# Playwright's resource types -> the RequestContext the table was fitted on.
#
# These vocabularies are not the same. The table's contexts come from HTTP
# Archive's content-type-derived `resource_type` (see RESOURCE_CONTEXT in
# llm-classifier/scripts/build_table.py), whereas Playwright reports the
# fetch destination. Two mappings are genuinely lossy and are called out
# because they shift which table entry answers:
#   media -> VIDEO   Playwright does not split audio from video.
#   xhr/fetch -> OTHER   build_table.py notes that the log's `json` rows land
#                        in OTHER, and XHR/fetch payloads are mostly JSON.
# Using the response's content-type would sharpen this, but a blocked request
# has no response, so the browser could not do that at block time either.
PLAYWRIGHT_TO_CONTEXT: dict[str, object] = {
    "document": RC.HTML,
    "stylesheet": RC.CSS,
    "image": RC.IMAGE,
    "media": RC.VIDEO,
    "font": RC.FONT,
    "script": RC.SCRIPT,
    "texttrack": RC.TEXT,
    "xhr": RC.OTHER,
    "fetch": RC.OTHER,
    "eventsource": RC.OTHER,
    "websocket": RC.OTHER,
    "manifest": RC.OTHER,
    "ping": RC.OTHER,
    "beacon": RC.OTHER,
    "other": RC.OTHER,
}


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().split(":")[0]


def is_third_party(request_url: str, page_url: str) -> bool:
    """Would Firefox treat this request as third-party to the page?

    ETP only blocks *third-party* trackers, but `is_tracker()` answers from the
    URL alone and so also flags a tracker vendor serving its own site. That
    distinction is not academic here: Tranco's head contains the ad-tech
    vendors themselves (media.net, optimizely.com, doubleverify.com,
    taboola.com), whose first-party assets Disconnect flags and ETP never
    touches.

    Suffix comparison stands in for a public-suffix lookup, which is sound
    because the crawl's page URLs are Tranco apex domains; it would need a real
    PSL if the page list carried arbitrary deep URLs.
    """
    h, p_ = _host(request_url), _host(page_url)
    if not h or not p_:
        return False
    return not (h == p_ or h.endswith("." + p_) or p_.endswith("." + h))


def _domains_by_owner() -> dict[str, set[str]]:
    """Owner -> every domain the Disconnect list associates with it.

    ETP does not block a tracker that belongs to the same entity as the site
    being visited, and that relationship crosses registrable domains: eBay
    serves images from ebayimg.com, Shein from ltwebstatic.com, Airbnb from
    muscache.com. Reconstructing the owner's domain set from the list lets us
    model that exemption instead of counting those requests as blockable.
    """
    out: dict[str, set[str]] = {}
    for e in disconnect.entries():
        owner = e.get("organization")
        if not owner:
            continue
        doms = out.setdefault(owner, set())
        doms.add(e["pattern"].lower().lstrip("."))
        owner_url = e.get("organization_url")
        if owner_url:
            h = _host(owner_url if "//" in owner_url else "http://" + owner_url)
            if h:
                doms.add(h)
    return out


def _related(a: str, b: str) -> bool:
    """Same host or one a subdomain of the other.

    Used instead of extracting a registrable domain, because "last two labels"
    turns amazon.co.uk into co.uk and would then call every .co.uk site the
    same entity.
    """
    return bool(a) and bool(b) and (a == b or a.endswith("." + b)
                                    or b.endswith("." + a))


def same_entity(request_url: str, page_url: str,
                by_owner: dict[str, set[str]]) -> bool:
    """Is this tracker owned by the same entity as the page it loaded on?"""
    owner = disconnect.tracker_owner(request_url)
    if not owner:
        return False
    page_host = _host(page_url)
    return any(_related(page_host, d) for d in by_owner.get(owner, ()))


def context_for(resource_type: str):
    return PLAYWRIGHT_TO_CONTEXT.get((resource_type or "").lower(), RC.OTHER)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, with ties averaged. Avoids a scipy dependency."""
    if len(xs) < 3:
        return None

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def load_type_log(arm_dir: Path, idx: int, slug: str) -> dict[str, str]:
    """URL -> Playwright resource_type for one page, first occurrence winning."""
    f = arm_dir / f"types_{idx:04d}_{slug}.json"
    if not f.exists():
        return {}
    try:
        with open(f) as fh:
            entries = json.load(fh)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for e in entries:
        out.setdefault(e.get("url", ""), e.get("resource_type") or "other")
    return out


def read_har_entries(har_path: Path) -> list[dict]:
    """Parse one HAR once; callers derive several things from the entries."""
    if not har_path.exists():
        return []
    try:
        with open(har_path) as f:
            return json.load(f).get("log", {}).get("entries", [])
    except Exception:
        return []


def requests_from_entries(entries: list[dict]) -> list[tuple[str, int]]:
    """Every (url, transfer_bytes), duplicates preserved."""
    out = []
    for entry in entries:
        url = entry.get("request", {}).get("url", "")
        if not url:
            continue
        size = entry.get("response", {}).get("_transferSize")
        out.append((url, 0 if size is None or size < 0 else int(size)))
    return out


def final_page_url(entries: list[dict], requested_url: str) -> str:
    """The top-level URL that actually loaded, following redirects.

    Party classification has to judge against the document that ended up
    loading, not the one we asked for. Tranco lists onelink.me, which 301s to
    www.appsflyer.com; scored against onelink.me, AppsFlyer's own assets look
    third-party and a proxy would "block" 14 MB that Firefox correctly allowed
    as first-party. The chain is followed through the HAR's `redirectURL`
    instead of being guessed at, so sub-frame navigations are not mistaken for
    the top-level document.
    """
    if not entries:
        return requested_url
    by_url = {e.get("request", {}).get("url", ""): e for e in entries}
    current = entries[0]
    url = current.get("request", {}).get("url", "") or requested_url
    for _ in range(10):  # redirect loops are possible; cap the walk
        nxt = (current.get("response", {}) or {}).get("redirectURL") or ""
        if not nxt or nxt not in by_url:
            break
        current = by_url[nxt]
        url = nxt
    return url


def load_har_requests(har_path: Path) -> list[tuple[str, int]]:
    """Every (url, transfer_bytes) in one HAR, duplicates preserved."""
    return requests_from_entries(read_har_entries(har_path))


def page_slugs(arm_dir: Path) -> dict[int, str]:
    """Crawl index -> on-disk slug, from the HAR filenames."""
    out: dict[int, str] = {}
    for f in arm_dir.glob("har_*.json"):
        stem = f.name[len("har_"):-len(".json")]
        num, _, slug = stem.partition("_")
        try:
            out[int(num)] = slug
        except ValueError:
            continue
    return out


def err_stats(pred: list[int], actual: list[int]) -> dict:
    """Per-request and aggregate error, both of which matter here.

    The dashboard reports a sum, so the aggregate ratio is the headline; the
    per-request spread is reported too, since a table that is right only on
    average can still be useless for any single request.
    """
    n = len(pred)
    if not n:
        return {"n": 0}
    abs_err = [abs(p - a) for p, a in zip(pred, actual)]
    sp, sa = sum(pred), sum(actual)
    return {
        "n": n,
        "predicted_bytes": sp,
        "observed_bytes": sa,
        "aggregate_ratio": round(sp / sa, 4) if sa else None,
        "aggregate_error_pct": round(100.0 * (sp - sa) / sa, 2) if sa else None,
        "mae_bytes": round(st.fmean(abs_err), 1),
        "median_abs_err_bytes": round(st.median(abs_err), 1),
        "median_predicted": round(st.median(pred), 1),
        "median_observed": round(st.median(actual), 1),
        "spearman": (round(s, 4)
                     if (s := _spearman([float(x) for x in pred],
                                        [float(x) for x in actual])) is not None
                     else None),
    }


def page_agg_stats(per_page: list[tuple[int, int]]) -> dict:
    """How often a per-page predicted total lands near the real total.

    This is the metric the paper reports, because the dashboard shows a
    rolled-up figure rather than individual requests.
    """
    ratios = [p / a for p, a in per_page if a > 0]
    if not ratios:
        return {"n_pages": 0}
    return {
        "n_pages": len(ratios),
        "within_10pct": round(
            sum(1 for r in ratios if 0.9 <= r <= 1.1) / len(ratios), 4),
        "within_25pct": round(
            sum(1 for r in ratios if 0.75 <= r <= 1.25) / len(ratios), 4),
        "within_50pct": round(
            sum(1 for r in ratios if 0.5 <= r <= 1.5) / len(ratios), 4),
        "median_ratio": round(st.median(ratios), 4),
        "median_abs_pct_err": round(
            st.median([abs(r - 1.0) * 100 for r in ratios]), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--blocked", required=True,
                    help="blocked_observed_bytes.csv from compare_tracking_arms.py")
    ap.add_argument("--paired", required=True,
                    help="paired_pages.csv from compare_tracking_arms.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    normal_dir, private_dir = Path(args.normal), Path(args.private)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pages that loaded cleanly in both arms; a page that failed on one side
    # has no meaningful comparison.
    clean_idx = set()
    page_url: dict[int, str] = {}
    for r in csv.DictReader(open(args.paired)):
        if (r["outcome_normal"].startswith("ok")
                and r["outcome_private"].startswith("ok")):
            clean_idx.add(int(r["idx"]))
            page_url[int(r["idx"])] = r["url"]
    print(f"clean paired pages: {len(clean_idx)}")

    # ---------------------------------------------------------------- (1)/(3)
    # Walk the control arm, flag trackers with Disconnect, and score the table
    # against the sizes those requests really had.
    slugs = page_slugs(normal_dir)
    dc_pred: list[int] = []
    dc_actual: list[int] = []
    dc_per_page: list[tuple[int, int]] = []
    dc_rows: list[dict] = []
    # Third-party subset: the population ETP could actually act on.
    tp_pred: list[int] = []
    tp_actual: list[int] = []
    tp_per_page: list[tuple[int, int]] = []
    # Third-party *and* not same-entity: the closest offline proxy for ETP.
    cand_pred: list[int] = []
    cand_actual: list[int] = []
    cand_per_page: list[tuple[int, int]] = []
    by_owner = _domains_by_owner()
    n_control_requests = 0
    control_bytes = 0

    for idx in sorted(clean_idx):
        slug = slugs.get(idx)
        if slug is None:
            continue
        types = load_type_log(normal_dir, idx, slug)
        entries = read_har_entries(normal_dir / f"har_{idx:04d}_{slug}.json")
        reqs = requests_from_entries(entries)
        n_control_requests += len(reqs)
        control_bytes += sum(s for _, s in reqs)
        if not reqs:
            continue

        # Judge party against the document that actually loaded, not the URL
        # we asked for; see final_page_url.
        this_page = final_page_url(entries, page_url.get(idx, ""))
        urls = [u for u, _ in reqs]
        flags = disconnect.is_tracker_many(urls, ETP_CATEGORIES)
        p_sum = a_sum = 0
        tp_p_sum = tp_a_sum = 0
        cd_p_sum = cd_a_sum = 0
        for (url, size), is_tr in zip(reqs, flags):
            if not is_tr:
                continue
            rtype = types.get(url, "other")
            est, _cpu_ms = llm_classifier.estimate_resources(
                url, context_for(rtype), _UNKNOWN_INITIATOR, _UNKNOWN_METHOD)
            third = is_third_party(url, this_page)
            dc_pred.append(est)
            dc_actual.append(size)
            p_sum += est
            a_sum += size
            candidate = third and not same_entity(url, this_page, by_owner)
            if third:
                tp_pred.append(est)
                tp_actual.append(size)
                tp_p_sum += est
                tp_a_sum += size
            if candidate:
                cand_pred.append(est)
                cand_actual.append(size)
                cd_p_sum += est
                cd_a_sum += size
            dc_rows.append({
                "page_idx": idx, "page_url": this_page,
                "url": url, "resource_type": rtype,
                "predicted_bytes": est, "observed_bytes": size,
                "third_party": int(third), "etp_candidate": int(candidate),
            })
        if p_sum or a_sum:
            dc_per_page.append((p_sum, a_sum))
        if tp_p_sum or tp_a_sum:
            tp_per_page.append((tp_p_sum, tp_a_sum))
        if cd_p_sum or cd_a_sum:
            cand_per_page.append((cd_p_sum, cd_a_sum))

    # ------------------------------------------------------------------- (2)
    # The same scoring, restricted to what Firefox actually blocked.
    etp_pred: list[int] = []
    etp_actual: list[int] = []
    etp_per_page: dict[int, list[int]] = {}
    n_etp_tracking = 0
    etp_rows: list[dict] = []
    # Question (5) prices *every* block, matched or not: the estimator needs
    # only the URL, and the page-level delta it is compared against was
    # measured over all the blocking.
    all_pred_by_page: dict[int, int] = {}
    cascade_pred_by_page: dict[int, int] = {}
    for r in csv.DictReader(open(args.blocked)):
        if r["protection"] != "tracking":
            continue
        idx = int(r["page_idx"])
        if idx not in clean_idx:
            continue
        n_etp_tracking += 1
        est, _cpu_ms = llm_classifier.estimate_resources(
            r["blocked_url"], context_for(r["resource_type"]),
            _UNKNOWN_INITIATOR, _UNKNOWN_METHOD)
        # And again with the cascade, which is the like-for-like comparison
        # against a page-level delta: see FOLLOWUP_BYTES_PER_REQUEST.
        est_cascade, _cpu2 = llm_classifier.estimate_resources(
            r["blocked_url"], context_for(r["resource_type"]),
            _UNKNOWN_INITIATOR, _UNKNOWN_METHOD, True)
        all_pred_by_page[idx] = all_pred_by_page.get(idx, 0) + est
        cascade_pred_by_page[idx] = cascade_pred_by_page.get(idx, 0) + est_cascade
        if r["match_kind"] == "unmatched" or r["observed_bytes"] == "":
            continue
        obs = int(r["observed_bytes"])
        etp_pred.append(est)
        etp_actual.append(obs)
        acc = etp_per_page.setdefault(idx, [0, 0])
        acc[0] += est
        acc[1] += obs
        etp_rows.append({
            "page_idx": idx, "page_url": r["page_url"],
            "url": r["blocked_url"], "resource_type": r["resource_type"],
            "predicted_bytes": est, "observed_bytes": obs,
            "match_kind": r["match_kind"],
        })

    # ------------------------------------------------------------------- (4)
    measured_page_delta = 0
    # (5) The same delta, page by page, kept apart by whether ETP blocked
    # anything there. Pages it never touched must show no saving, so whatever
    # they do show is crawl-to-crawl churn and is the drift estimate.
    delta_blocked: dict[int, int] = {}
    delta_untouched: list[int] = []
    for r in csv.DictReader(open(args.paired)):
        idx = int(r["idx"])
        if idx not in clean_idx:
            continue
        saved = int(r["bytes_saved"])
        measured_page_delta += saved
        if int(r["n_blocked_tracking"]) > 0:
            delta_blocked[idx] = saved
        else:
            delta_untouched.append(saved)

    # Only pages where both sides of the comparison exist: ETP blocked
    # something (so there is a prediction) and the page loaded cleanly in both
    # arms (so there is a delta).
    q5_idx = sorted(set(delta_blocked) & set(all_pred_by_page))
    q5_pred = sum(all_pred_by_page[i] for i in q5_idx)
    q5_pred_cascade = sum(cascade_pred_by_page[i] for i in q5_idx)
    q5_measured = sum(delta_blocked[i] for i in q5_idx)
    drift_per_page = (st.fmean(delta_untouched) if delta_untouched else 0.0)
    q5_measured_corrected = q5_measured - drift_per_page * len(q5_idx)
    q5_per_page = [(all_pred_by_page[i], delta_blocked[i]) for i in q5_idx]

    report = {
        "inputs": {
            "clean_paired_pages": len(clean_idx),
            "control_requests": n_control_requests,
            "control_transfer_bytes": control_bytes,
            "etp_categories": ETP_CATEGORIES,
            "estimator": "llm_classifier (table build)",
        },
        "q1_size_accuracy_disconnect_population": err_stats(dc_pred, dc_actual),
        "q1_per_page_aggregate": page_agg_stats(dc_per_page),
        "q1b_size_accuracy_disconnect_third_party_only":
            err_stats(tp_pred, tp_actual),
        "q1b_per_page_aggregate": page_agg_stats(tp_per_page),
        "q1c_size_accuracy_etp_candidate_proxy":
            err_stats(cand_pred, cand_actual),
        "q1c_per_page_aggregate": page_agg_stats(cand_per_page),
        "q1d_first_party_excluded": {
            "n_requests": len(dc_pred) - len(tp_pred),
            "observed_bytes": sum(dc_actual) - sum(tp_actual),
            "note": "Disconnect-listed but first-party, so ETP never blocks "
                    "them. Tranco's head includes ad-tech vendors' own sites, "
                    "which is where most of this comes from.",
        },
        "q2_size_accuracy_etp_blocked": err_stats(etp_pred, etp_actual),
        "q2_per_page_aggregate": page_agg_stats(
            [(v[0], v[1]) for v in etp_per_page.values()]),
        "q3_classification_divergence": {
            "disconnect_flagged_in_control": len(dc_pred),
            "disconnect_flagged_third_party": len(tp_pred),
            "disconnect_third_party_cross_entity": len(cand_pred),
            "etp_blocked_tracking": n_etp_tracking,
            "count_ratio_disconnect_over_etp": (
                round(len(dc_pred) / n_etp_tracking, 3)
                if n_etp_tracking else None),
            "count_ratio_third_party_over_etp": (
                round(len(tp_pred) / n_etp_tracking, 3)
                if n_etp_tracking else None),
            "count_ratio_candidate_over_etp": (
                round(len(cand_pred) / n_etp_tracking, 3)
                if n_etp_tracking else None),
            "disconnect_flagged_observed_bytes": sum(dc_actual),
            "etp_blocked_observed_bytes_matched": sum(etp_actual),
            "note": "Different lists. Disconnect is scored on the control arm "
                    "where every request has a size; ETP's set is what Firefox "
                    "refused, and only the matched subset has an observed size, "
                    "so its byte total is a lower bound.",
        },
        "q5_blocked_estimate_vs_page_delta": {
            "n_pages": len(q5_idx),
            "n_blocks_priced": n_etp_tracking,
            "predicted_blocked_bytes": q5_pred,
            "predicted_blocked_bytes_with_followups": q5_pred_cascade,
            "measured_page_delta_bytes": q5_measured,
            "measured_page_delta_bytes_drift_corrected":
                round(q5_measured_corrected),
            "drift_per_untouched_page_bytes": round(drift_per_page),
            "n_untouched_pages": len(delta_untouched),
            "ratio_predicted_over_measured": (
                round(q5_pred / q5_measured, 3) if q5_measured else None),
            "ratio_predicted_over_measured_drift_corrected": (
                round(q5_pred / q5_measured_corrected, 3)
                if q5_measured_corrected else None),
            "ratio_with_followups_over_measured_drift_corrected": (
                round(q5_pred_cascade / q5_measured_corrected, 3)
                if q5_measured_corrected else None),
            "n_pages_predicted_exceeds_measured": sum(
                1 for pr, ms in q5_per_page if pr > ms),
            "n_pages_measured_negative": sum(
                1 for _pr, ms in q5_per_page if ms < 0),
            "median_predicted_bytes_per_page": (
                round(st.median([pr for pr, _ in q5_per_page]))
                if q5_per_page else None),
            "median_measured_bytes_per_page": (
                round(st.median([ms for _, ms in q5_per_page]))
                if q5_per_page else None),
            "note": "Predicted size of every request ETP blocked (unmatched "
                    "ones included -- the estimator needs only the URL) "
                    "against the measured transfer delta of the pages those "
                    "blocks happened on. Two ratios, because the estimator "
                    "answers two questions: priced per request, it covers "
                    "only what was refused and should sit well below 1, "
                    "since the delta also contains the subresources a "
                    "blocked tracker never got to request; priced with "
                    "include_followups it estimates those too and should sit "
                    "near 1. Pages ETP never touched give the drift "
                    "estimate, since their delta is churn by construction.",
        },
        "q4_end_to_end": {
            "predicted_savings_bytes": sum(dc_pred),
            "predicted_savings_bytes_third_party_only": sum(tp_pred),
            "predicted_savings_bytes_etp_candidate_proxy": sum(cand_pred),
            "measured_direct_blocked_bytes_lower_bound": sum(etp_actual),
            "measured_page_level_delta_bytes": measured_page_delta,
            "note": "The page-level delta includes subresources a blocked "
                    "tracker never got to request; the table does not model "
                    "those, so it is not the target it should be judged "
                    "against.",
        },
    }

    with open(out_dir / "estimate_vs_etp.json", "w") as f:
        json.dump(report, f, indent=2)
    for name, rows, cols in (
        ("estimate_vs_observed_disconnect.csv", dc_rows,
         ["page_idx", "page_url", "url", "resource_type",
          "predicted_bytes", "observed_bytes", "third_party",
          "etp_candidate"]),
        ("estimate_vs_observed_etp_blocked.csv", etp_rows,
         ["page_idx", "page_url", "url", "resource_type",
          "predicted_bytes", "observed_bytes", "match_kind"]),
    ):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def show(title: str, s: dict, agg: dict) -> None:
        print()
        print(f"--- {title}")
        if not s.get("n"):
            print("    no rows")
            return
        print(f"    requests scored     {s['n']:,}")
        print(f"    predicted total     {s['predicted_bytes']/1e6:,.1f} MB")
        print(f"    observed total      {s['observed_bytes']/1e6:,.1f} MB")
        print(f"    aggregate error     {s['aggregate_error_pct']:+.1f}%  "
              f"(ratio {s['aggregate_ratio']:.3f})")
        print(f"    per-request MAE     {s['mae_bytes']:,.0f} B   "
              f"median abs err {s['median_abs_err_bytes']:,.0f} B")
        print(f"    median pred/obs     {s['median_predicted']:,.0f} B "
              f"vs {s['median_observed']:,.0f} B")
        if s["spearman"] is not None:
            print(f"    Spearman            {s['spearman']:.3f}")
        if agg.get("n_pages"):
            print(f"    per-page totals     within10%="
                  f"{agg['within_10pct']:.1%}  within25%={agg['within_25pct']:.1%}"
                  f"  median|err|={agg['median_abs_pct_err']:.1f}%  "
                  f"(n={agg['n_pages']} pages)")

    print()
    print("=== llm-classifier table vs real ETP outcome ===")
    print(f"control arm: {n_control_requests:,} requests, "
          f"{control_bytes/1e6:,.1f} MB over {len(clean_idx)} pages")

    show("(1) size accuracy, Disconnect-flagged control requests",
         report["q1_size_accuracy_disconnect_population"],
         report["q1_per_page_aggregate"])
    show("(1b) same, third-party only -- the population ETP can act on",
         report["q1b_size_accuracy_disconnect_third_party_only"],
         report["q1b_per_page_aggregate"])
    show("(1c) same, third-party and cross-entity -- closest ETP proxy",
         report["q1c_size_accuracy_etp_candidate_proxy"],
         report["q1c_per_page_aggregate"])
    fp = report["q1d_first_party_excluded"]
    print(f"    excluded as first-party: {fp['n_requests']:,} requests, "
          f"{fp['observed_bytes']/1e6:,.1f} MB")
    show("(2) size accuracy, exactly what ETP blocked",
         report["q2_size_accuracy_etp_blocked"],
         report["q2_per_page_aggregate"])

    d = report["q3_classification_divergence"]
    print()
    print("--- (3) classification divergence")
    print(f"    Disconnect flagged in control  {d['disconnect_flagged_in_control']:,}")
    print(f"      of which third-party         {d['disconnect_flagged_third_party']:,}")
    print(f"      and cross-entity             {d['disconnect_third_party_cross_entity']:,}")
    print(f"    ETP actually blocked           {d['etp_blocked_tracking']:,}")
    print(f"    ratio, all flagged / ETP       "
          f"{d['count_ratio_disconnect_over_etp']:.2f}x")
    print(f"    ratio, third-party / ETP       "
          f"{d['count_ratio_third_party_over_etp']:.2f}x")
    print(f"    ratio, cross-entity / ETP      "
          f"{d['count_ratio_candidate_over_etp']:.2f}x")

    e = report["q4_end_to_end"]
    print()
    print("--- (4) end-to-end savings estimate")
    print(f"    predicted savings, all flagged "
          f"{e['predicted_savings_bytes']/1e6:,.1f} MB")
    print(f"    predicted savings, third-party "
          f"{e['predicted_savings_bytes_third_party_only']/1e6:,.1f} MB")
    print(f"    predicted savings, ETP proxy   "
          f"{e['predicted_savings_bytes_etp_candidate_proxy']/1e6:,.1f} MB")
    print(f"    measured blocked bytes         "
          f"{e['measured_direct_blocked_bytes_lower_bound']/1e6:,.1f} MB "
          f"(lower bound)")
    print(f"    measured page-level delta      "
          f"{e['measured_page_level_delta_bytes']/1e6:,.1f} MB "
          f"(includes cascades)")

    q = report["q5_blocked_estimate_vs_page_delta"]
    print()
    print("--- (5) estimated bytes of what ETP blocked vs measured page delta")
    print(f"    pages with blocks              {q['n_pages']:,} "
          f"({q['n_blocks_priced']:,} blocked requests priced)")
    print(f"    predicted blocked bytes        "
          f"{q['predicted_blocked_bytes']/1e6:,.1f} MB")
    print(f"    measured delta on those pages  "
          f"{q['measured_page_delta_bytes']/1e6:,.1f} MB")
    print(f"    drift correction               "
          f"{q['drift_per_untouched_page_bytes']/1e3:,.1f} kB/page over "
          f"{q['n_untouched_pages']:,} untouched pages "
          f"-> {q['measured_page_delta_bytes_drift_corrected']/1e6:,.1f} MB")
    print(f"    with follow-ups                "
          f"{q['predicted_blocked_bytes_with_followups']/1e6:,.1f} MB")
    if q["ratio_predicted_over_measured"] is not None:
        print(f"    ratio predicted / measured     "
              f"{q['ratio_predicted_over_measured']:.2f}x raw, "
              f"{q['ratio_predicted_over_measured_drift_corrected']:.2f}x "
              f"drift-corrected")
        print(f"    same, with follow-ups          "
              f"{q['ratio_with_followups_over_measured_drift_corrected']:.2f}x "
              f"drift-corrected")
    print(f"    pages where predicted > measured "
          f"{q['n_pages_predicted_exceeds_measured']:,} of {q['n_pages']:,} "
          f"({q['n_pages_measured_negative']:,} have a negative delta)")
    print(f"    median per page                 "
          f"{q['median_predicted_bytes_per_page']/1e3:,.1f} kB predicted vs "
          f"{q['median_measured_bytes_per_page']/1e3:,.1f} kB measured")
    print()
    print(f"Wrote {out_dir/'estimate_vs_etp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
