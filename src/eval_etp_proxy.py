"""
Score the offline ETP proxy against what Firefox really blocked, then redo the
savings estimate with it.

The paired crawl gives a labelled set the proxy can be graded on directly: for
a request the control arm fetched, the blocking arm says whether Firefox
refused the same request. So "would ETP block this?" becomes a plain
classification problem with ground truth.

Only requests that appear in *both* arms are scored. A URL seen in one load and
not the other says nothing about the classifier -- ad slots rotate and
cache-busting query strings churn -- so counting those would measure page
variance rather than classification. URLs are compared with the query stripped,
for the same reason.

Two classifiers are graded side by side:

  naive       `disconnect.is_tracker(url, "Advertising,Analytics,Social")`,
              the URL-only lookup, which is what an offline estimate reaches
              for first
  proxy       src/firefox_etp_proxy.py, which adds third-party, entity
              allowlist and exception handling on top of the same list data

and the savings estimate is recomputed under each, so the cost of getting
classification wrong is expressed in megabytes rather than in F1.

Usage:
    python src/eval_etp_proxy.py \\
        --normal   data/raw/firefox_crawl_500_tracking/normal \\
        --private  data/raw/firefox_crawl_500_tracking/private \\
        --paired   data/raw/firefox_crawl_500_tracking/paired_pages.csv \\
        --out      data/raw/firefox_crawl_500_tracking
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import disconnect
import llm_classifier

from compare_estimate_vs_etp import (ETP_CATEGORIES, _UNKNOWN_INITIATOR,
                                     _UNKNOWN_METHOD, context_for,
                                     final_page_url, load_type_log,
                                     page_slugs, read_har_entries,
                                     requests_from_entries)
from firefox_etp_proxy import FirefoxETPProxy

TRACKING_MARKERS = ("NS_ERROR_TRACKING_URI", "NS_ERROR_SOCIALTRACKING_URI",
                    "NS_ERROR_EMAILTRACKING_URI")


def strip_query(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}".lower()


def private_arm_state(private_dir: Path, idx: int,
                      slug: str) -> tuple[set[str], set[str]]:
    """(blocked, seen) stripped-URL sets for one page of the blocking arm."""
    blocked: set[str] = set()
    f = private_dir / f"blocked_{idx:04d}_{slug}.json"
    if f.exists():
        try:
            for b in json.loads(f.read_text()).get("blocked", []):
                if any(m in b.get("failure", "") for m in TRACKING_MARKERS):
                    blocked.add(strip_query(b.get("url", "")))
        except Exception:
            pass
    seen = {strip_query(u) for u in load_type_log(private_dir, idx, slug)}
    return blocked, seen


class Score:
    """Confusion matrix plus the bytes riding on each cell."""

    def __init__(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0
        self.fp_bytes = 0
        self.fn_bytes = 0
        self.fp_hosts: Counter = Counter()

    def add(self, predicted: bool, actual: bool, size: int, host: str) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
            self.fp_bytes += size
            self.fp_hosts[host] += size
        elif not predicted and actual:
            self.fn += 1
            self.fn_bytes += size
        else:
            self.tn += 1

    def as_dict(self) -> dict:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "false_positive_bytes": self.fp_bytes,
            "false_negative_bytes": self.fn_bytes,
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lists", default=None,
                    help="Directory of Firefox lists (default: "
                         "data/external/firefox_lists).")
    ap.add_argument("--pbm", action="store_true",
                    help="Model a real Private Window, honouring "
                         "isPrivateBrowsingOnly exceptions. The crawl's arm was "
                         "a normal window with tracking protection forced on, "
                         "so leave this off to match it.")
    args = ap.parse_args()

    normal_dir, private_dir = Path(args.normal), Path(args.private)
    out_dir = Path(args.out)
    proxy = (FirefoxETPProxy(args.lists, pbm=args.pbm) if args.lists
             else FirefoxETPProxy(pbm=args.pbm))
    print("proxy:", json.dumps(proxy.describe()))

    clean: dict[int, str] = {}
    measured_delta = 0
    # Split page deltas by whether ETP actually blocked anything there. Pages
    # where it blocked nothing should show no saving, so whatever they do show
    # is drift between the two crawls -- ad rotation, changed content -- and is
    # the right control for the pages where it did block.
    delta_blocked: list[int] = []
    delta_unblocked: list[int] = []
    for r in csv.DictReader(open(args.paired)):
        if (r["outcome_normal"].startswith("ok")
                and r["outcome_private"].startswith("ok")):
            clean[int(r["idx"])] = r["url"]
            saved = int(r["bytes_saved"])
            measured_delta += saved
            if int(r["n_blocked_tracking"]) > 0:
                delta_blocked.append(saved)
            else:
                delta_unblocked.append(saved)
    print(f"clean paired pages: {len(clean)}")

    slugs_n, slugs_p = page_slugs(normal_dir), page_slugs(private_dir)
    naive, prox = Score(), Score()
    est_naive = est_proxy = 0
    n_naive_flagged = n_proxy_flagged = 0
    obs_proxy_flagged = 0
    n_scored = n_skipped = 0
    reasons: Counter = Counter()

    for idx, page in sorted(clean.items()):
        sn, sp = slugs_n.get(idx), slugs_p.get(idx)
        if not sn or not sp:
            continue
        types = load_type_log(normal_dir, idx, sn)
        entries = read_har_entries(normal_dir / f"har_{idx:04d}_{sn}.json")
        reqs = requests_from_entries(entries)
        if not reqs:
            continue
        # Redirects move the top-level document, and party depends on where it
        # landed; see final_page_url.
        page = final_page_url(entries, page)
        blocked, seen = private_arm_state(private_dir, idx, sp)

        urls = [u for u, _ in reqs]
        naive_flags = disconnect.is_tracker_many(urls, ETP_CATEGORIES)

        for (url, size), naive_is in zip(reqs, naive_flags):
            p_block, why = proxy.decide(url, page)
            rtype = types.get(url, "other")

            # Savings estimate under each classifier.
            if naive_is:
                n_naive_flagged += 1
                est_naive += llm_classifier.estimate_resources(
                    url, context_for(rtype),
                    _UNKNOWN_INITIATOR, _UNKNOWN_METHOD)[0]
            if p_block:
                n_proxy_flagged += 1
                est_proxy += llm_classifier.estimate_resources(
                    url, context_for(rtype),
                    _UNKNOWN_INITIATOR, _UNKNOWN_METHOD)[0]
                obs_proxy_flagged += size
            elif why != "not-listed":
                # Only interesting for requests that *are* on the tracker list:
                # these are the ones a naive lookup would have counted.
                reasons[why] += 1

            # Grade only where the blocking arm actually re-requested this URL.
            key = strip_query(url)
            if key in blocked:
                actual = True
            elif key in seen:
                actual = False
            else:
                n_skipped += 1
                continue
            n_scored += 1
            host = urlsplit(url).netloc.lower()
            naive.add(bool(naive_is), actual, size, host)
            prox.add(p_block, actual, size, host)

    drift_per_page = (sum(delta_unblocked) / len(delta_unblocked)
                      if delta_unblocked else 0.0)
    measured_on_blocked_pages = sum(delta_blocked)
    measured_did = measured_on_blocked_pages - drift_per_page * len(delta_blocked)

    report = {
        "proxy_config": proxy.describe(),
        "scored_requests": n_scored,
        "skipped_not_in_both_arms": n_skipped,
        "classification": {"naive_is_tracker": naive.as_dict(),
                           "firefox_etp_proxy": prox.as_dict()},
        "flagged_counts": {"naive": n_naive_flagged, "proxy": n_proxy_flagged},
        "savings_estimate_bytes": {
            "naive_classification": est_naive,
            "proxy_classification": est_proxy,
            "observed_bytes_of_proxy_flagged": obs_proxy_flagged,
            "measured_page_level_delta": measured_delta,
            "measured_on_pages_with_blocks": measured_on_blocked_pages,
            "measured_drift_corrected": round(measured_did),
        },
        "page_delta_decomposition": {
            "n_pages_with_blocks": len(delta_blocked),
            "n_pages_without_blocks": len(delta_unblocked),
            "delta_on_pages_with_blocks": measured_on_blocked_pages,
            "delta_on_pages_without_blocks": sum(delta_unblocked),
            "drift_per_page_bytes": round(drift_per_page),
            "note": "Pages ETP did not touch still differ -- that is "
                    "drift, not saving. Subtracting it from the pages it did "
                    "touch gives the drift-corrected figure.",
        },
        "proxy_block_reasons": dict(reasons),
        "top_false_positive_hosts_by_bytes":
            [{"host": h, "bytes": b} for h, b in prox.fp_hosts.most_common(15)],
    }
    (out_dir / "etp_proxy_eval.json").write_text(json.dumps(report, indent=2))

    print()
    print("=== classification vs what Firefox really blocked ===")
    print(f"scored {n_scored:,} requests present in both arms "
          f"({n_skipped:,} skipped as non-recurring)")
    print(f"{'':<22}{'precision':>10}{'recall':>9}{'F1':>8}"
          f"{'FP':>8}{'FN':>7}{'FP MB':>9}")
    for name, s in (("naive is_tracker", naive.as_dict()),
                    ("firefox_etp_proxy", prox.as_dict())):
        print(f"{name:<22}{s['precision']:>10.3f}{s['recall']:>9.3f}"
              f"{s['f1']:>8.3f}{s['fp']:>8,}{s['fn']:>7,}"
              f"{s['false_positive_bytes']/1e6:>9.1f}")

    e = report["savings_estimate_bytes"]
    print()
    print("=== savings estimate, same size table, different classifier ===")
    print(f"naive classification   {n_naive_flagged:6,} requests  "
          f"{e['naive_classification']/1e6:7.1f} MB predicted")
    print(f"proxy classification   {n_proxy_flagged:6,} requests  "
          f"{e['proxy_classification']/1e6:7.1f} MB predicted")
    print(f"  observed bytes of those same requests   "
          f"{e['observed_bytes_of_proxy_flagged']/1e6:7.1f} MB")
    d = report["page_delta_decomposition"]
    print()
    print("=== measured savings, corrected for crawl-to-crawl drift ===")
    print(f"raw net page-level delta, all {len(clean)} pages   "
          f"{e['measured_page_level_delta']/1e6:7.1f} MB")
    print(f"  on the {d['n_pages_with_blocks']} pages ETP blocked on      "
          f"{d['delta_on_pages_with_blocks']/1e6:7.1f} MB")
    print(f"  on the {d['n_pages_without_blocks']} pages it did not       "
          f"{d['delta_on_pages_without_blocks']/1e6:7.1f} MB  "
          f"<- pure drift, should be ~0")
    print(f"  drift-corrected measured savings        "
          f"{e['measured_drift_corrected']/1e6:7.1f} MB")
    print()
    for label, target in (("raw net delta", e["measured_page_level_delta"]),
                          ("blocked pages only",
                           e["measured_on_pages_with_blocks"]),
                          ("drift-corrected", e["measured_drift_corrected"])):
        ok = e["proxy_classification"] <= target
        print(f"  predicted {e['proxy_classification']/1e6:.1f} MB <= "
              f"{label} {target/1e6:.1f} MB ? "
              f"{'YES' if ok else 'NO'}")
    print()
    print("why the proxy let things through (counts over all control requests):")
    for why, n in Counter(reasons).most_common():
        print(f"    {why:<22} {n:,}")
    print()
    print(f"Wrote {out_dir/'etp_proxy_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
