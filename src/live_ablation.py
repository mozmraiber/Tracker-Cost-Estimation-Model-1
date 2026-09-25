"""Measure a tracker's follow-up bytes by ablating it from a *live* page.

THE OTHER HALF OF `src/replay_ablation.py`
------------------------------------------
Both scripts ask the same question -- what does the page stop fetching when
one tracker host is taken away -- and they trade the same two properties
against each other.

Replay freezes the responses, so the load is deterministic and the delta is
exact, but a frozen ad response is a stale one: the auction does not run, the
creative URL the script mints on this impression is not in the recording, and
the subtree that never had a fixed URL is invisible. Its number is a floor.

This script keeps the network live. Every response is real, the auction runs,
the creative loads, and nothing is missing by construction. What comes back
instead is the noise that replay removed -- a live page load differs from the
next one by first-party media, lazy content and whatever the ad server felt
like serving -- so the delta has to be averaged over repeats and is only
readable if the accounting is narrow enough to keep the churn out.

TWO ARMS, BOTH INTERCEPTED
--------------------------
The ablation is a `route` handler that aborts one host. Registering one makes
Playwright intercept *every* request to evaluate the predicate, which costs
time the page can notice: on forbes.com that alone moved the page total by
about 9 kB, which `src/replay_ablation.py` first read as a phantom cascade for
three unrelated hosts. So the control arm is not an un-intercepted load. It
registers the identical handler against `PLACEBO_HOST`, a name no page
resolves, and the two arms differ only in which host is named.

Arms alternate inside each repeat, control-first on even repeats and
ablated-first on odd ones, so a site that slows down over the run cannot pass
that drift off as a cascade.

THREE WIDTHS OF ACCOUNTING
--------------------------
The same paired difference is reported over three sets of requests, because
the choice of set is the whole measurement problem:

  page         every request. What `paired_pages.csv` calls `bytes_saved`.
               Complete and unusable on its own: 1.37 MB of churn a page
               against a cascade of tens of kB.

  third-party  every request to a host that is not the page's own. This is
               the one this script exists for. It drops the first-party
               media where nearly all the churn lives, and unlike
               `bytes_saved_tracking` it *keeps* the ad creatives, iframes
               and CDNs that no list names -- which is exactly the part the
               shipped 47 kB is a lower bound because it cannot see.

  listed       Disconnect-matched requests only, i.e. `bytes_saved_tracking`.
               Reported so this instrument can be read against the one the
               constant currently comes from.

A/A CONTROL
-----------
`--aa` runs the same paired design with the placebo host on *both* sides, so
the two arms are identical and every difference it reports is churn. That is
the resolution floor: a cascade smaller than the A/A spread at a given width
has not been measured, whatever the mean says. Run it before believing
anything below.

USAGE
-----
    python src/live_ablation.py --url https://www.nbcnews.com \\
        --host securepubads.g.doubleclick.net --reps 6 --aa

    python src/live_ablation.py --url https://www.nbcnews.com \\
        --auto-top 3 --reps 6
"""

from __future__ import annotations

import argparse
import collections
import csv
import functools
import json
import math
import statistics as st
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from compare_estimate_vs_etp import is_third_party  # noqa: E402
from firefox_crawl_500_tracking import build_prefs  # noqa: E402
from replay_ablation import (CASCADING_TYPES, PLACEBO_HOST,  # noqa: E402
                             is_tracker, slugify)

WIDTHS = ("page", "third_party", "listed")

#: The ten-pass crawl's blocked requests, for `etp_blocked_hosts`.
BLOCKED_CSV = (ROOT / "data/raw/firefox_crawl_500_tracking_x10"
               / "blocked_observed_bytes.csv")


def host_of(url: str) -> str:
    return urlsplit(url).netloc


def blocks(*hosts: str):
    """Predicate matching any of `hosts` and anything under them."""
    def pred(u: str) -> bool:
        h = host_of(u)
        return any(h == x or h.endswith("." + x) for x in hosts)
    return pred


# --------------------------------------------------------------------------


def load(browser, url: str, *ablate_hosts: str,
         timeout_s: int, settle_s: float) -> dict:
    """One live load with `ablate_host` aborted, returning its byte accounting.

    `ablate_hosts` is never empty: the control arm passes `PLACEBO_HOST` so
    that both arms pay the same interception cost. See the module docstring.
    More than one host is the joint arm -- see `paired`.
    """
    aborted: list[str] = []
    atypes: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        har_path = Path(tmp) / "page.har"
        ctx = browser.new_context(
            record_har_path=str(har_path),
            record_har_content="omit",
            record_har_mode="full",
        )

        def _abort(route):
            aborted.append(route.request.url)
            # The resource type of what we removed, not just how much of it.
            # `FOLLOWUP_BYTES_PER_REQUEST` is charged per *cascading* blocked
            # request -- only code cascades -- and its denominator counts
            # those alone. Counting a host's beacons here too would divide the
            # same cascade by a bigger number and read low, which is exactly
            # what the first wide run did: 14.6 KB a request against the
            # crawl's 47.0, on a denominator four times too large.
            atypes.append(route.request.resource_type)
            route.abort()

        ctx.route(blocks(*ablate_hosts), _abort)
        page = ctx.new_page()
        types: dict[str, str] = {}
        page.on("request",
                lambda r: types.setdefault(r.url, r.resource_type))
        status = "ok"
        try:
            page.goto(url, wait_until="load", timeout=timeout_s * 1000)
        except Exception as exc:
            status = f"goto: {type(exc).__name__}"
        page.wait_for_timeout(int(settle_s * 1000))
        ctx.close()
        har = json.loads(har_path.read_text()) if har_path.exists() else {}

    obs: list[tuple[str, int]] = []
    for e in har.get("log", {}).get("entries", []):
        u = e.get("request", {}).get("url", "")
        if not u:
            continue
        size = e.get("response", {}).get("_transferSize")
        obs.append((u, 0 if size is None or size < 0 else int(size)))

    widths = {"page": 0, "third_party": 0, "listed": 0}
    host_bytes: dict[str, int] = {}
    for u, size in obs:
        widths["page"] += size
        if is_third_party(u, url):
            widths["third_party"] += size
            if is_tracker(u):
                widths["listed"] += size
        host_bytes[host_of(u)] = host_bytes.get(host_of(u), 0) + size
    return {
        "status": status, "n_requests": len(obs), "n_aborted": len(aborted),
        "n_aborted_casc": sum(1 for t in atypes if t in CASCADING_TYPES),
        # The full histogram, because "cascading" is defined differently in
        # the two places this has to be comparable with. This file has always
        # meant script/document/xhr/fetch -- anything that can pull something
        # in -- while `FOLLOWUP_BYTES_PER_REQUEST` is charged only over the
        # contexts that run code, `CASCADING = ("SCRIPT", "HTML")` in
        # `fit_followups.py`. A tracker's beacons are xhr and fetch and there
        # are a lot of them, so the wider definition divides the same cascade
        # by a much bigger number and reads low against the crawl's 47.0 KB.
        # Keeping the counts lets the analysis use either denominator.
        "aborted_types": dict(collections.Counter(atypes)),
        "bytes": widths, "host_bytes": host_bytes, "types": types,
    }


def paired(browser, url: str, *hosts: str, reps: int,
           timeout_s: int, settle_s: float, quiet: bool = False) -> dict:
    """`reps` paired loads of `url`, one arm ablating `host`.

    Returns the per-repeat differences at each width. The target's own bytes
    are taken from the control arm of the *same* repeat rather than from a
    pooled average, because a host that no-showed on one visit should not be
    charged for it on another.
    """
    host = hosts[0] if len(hosts) == 1 else "|".join(hosts)
    diffs: dict[str, list[float]] = {w: [] for w in WIDTHS}
    own: list[int] = []
    rows = []
    for i in range(reps):
        order = ["control", "ablated"] if i % 2 == 0 else ["ablated", "control"]
        arms = {}
        for arm in order:
            targets = (PLACEBO_HOST,) if arm == "control" else hosts
            arms[arm] = load(browser, url, *targets,
                             timeout_s=timeout_s, settle_s=settle_s)
        c, a = arms["control"], arms["ablated"]
        # A host can serve from several subdomains; charge them all, the same
        # way the ablation predicate removes them all. The joint arm charges
        # every host it removed.
        own_i = sum(v for h, v in c["host_bytes"].items()
                    if any(h == x or h.endswith("." + x) for x in hosts))
        own.append(own_i)
        for w in WIDTHS:
            # `own_i` is subtracted at every width: a tracker host's own bytes
            # are third-party and listed as well as on the page, so all three
            # totals contain them and all three must have them taken out to
            # leave the subtree. The exception is a host whose URLs the list
            # matches only some of, where the listed width takes out slightly
            # too much; `third_party` is the width to read.
            diffs[w].append(c["bytes"][w] - a["bytes"][w] - own_i)
        # Where the bytes moved, not just how many. A page can *gain*
        # traffic when a tracker is taken away -- an ad slot refilled from
        # somewhere else, a widget's space given to lazy content -- and a
        # single signed total cannot tell that from a mismeasurement.
        moved = {h: c["host_bytes"].get(h, 0) - a["host_bytes"].get(h, 0)
                 for h in set(c["host_bytes"]) | set(a["host_bytes"])}
        top = dict(sorted(moved.items(), key=lambda kv: -abs(kv[1]))[:12])
        rows.append({"rep": i, "order": order, "own": own_i,
                     "control": c["bytes"], "ablated": a["bytes"],
                     "n_aborted": a["n_aborted"],
                     "n_aborted_casc": a["n_aborted_casc"], "moved": top})
        if not quiet:
            print(f"    rep {i}: own {own_i/1e3:7.1f} kB  "
                  + "  ".join(f"{w} {diffs[w][-1]/1e3:+8.1f}" for w in WIDTHS)
                  + f"   ({a['n_aborted']} aborted)")
    return {"host": host, "reps": rows, "diffs": diffs, "own": own}


def summarise(diffs: dict[str, list[float]]) -> dict:
    out = {}
    for w, xs in diffs.items():
        n = len(xs)
        sd = st.stdev(xs) if n > 1 else float("nan")
        out[w] = {"n": n, "mean": st.mean(xs) if n else float("nan"),
                  "sd": sd, "se": sd / math.sqrt(n) if n > 1 else float("nan")}
    return out


def report(name: str, s: dict) -> None:
    print(f"  {name}")
    for w in WIDTHS:
        v = s[w]
        print(f"    {w:12s} {v['mean']/1e3:+9.1f} kB  "
              f"+-{v['se']/1e3:7.1f} (sd {v['sd']/1e3:7.1f}, n={v['n']})")


# --------------------------------------------------------------------------


@functools.cache
def etp_blocked_hosts() -> frozenset[str]:
    """Hosts Firefox's ETP actually refused, from the ten-pass crawl.

    `is_tracker` answers from the Disconnect list, and Firefox's ETP tracking
    tables are not the Disconnect list. Selecting candidates by the former
    picks hosts the shipped estimator is never asked about: the list names
    OneTrust and Osano, ETP blocked a consent manager zero times in 18,891
    blocked requests, and those two supplied the largest and most reproducible
    "cascades" in the first 21-page run -- +3.6 MB on braze.com, -634 KB on
    ted.com -- while being gates rather than subtrees.

    Empty when the crawl is not on disk, in which case `recon` falls back to
    the list and says so.
    """
    if not BLOCKED_CSV.exists():
        return frozenset()
    with open(BLOCKED_CSV, newline="") as f:
        return frozenset(host_of(r["blocked_url"]).lower()
                         for r in csv.DictReader(f)
                         if r.get("protection") == "tracking")


def recon(browser, url: str, top: int, *, timeout_s: int,
          settle_s: float) -> list[str]:
    """One un-ablated load, to pick the hosts worth spending repeats on.

    Candidates are hosts ETP is known to block, not merely hosts the
    Disconnect list names; see `etp_blocked_hosts`.
    """
    r = load(browser, url, PLACEBO_HOST, timeout_s=timeout_s,
             settle_s=settle_s)
    blocked = etp_blocked_hosts()
    cands: dict[str, int] = {}
    for u, t in r["types"].items():
        if not (is_tracker(u) and is_third_party(u, url)):
            continue
        if t not in CASCADING_TYPES:
            continue
        h = host_of(u)
        if blocked and h.lower() not in blocked:
            continue
        cands[h] = max(cands.get(h, 0), r["host_bytes"].get(h, 0))
    return [h for h, _ in sorted(cands.items(), key=lambda kv: -kv[1])[:top]]


def measure_page(browser, url: str, *, hosts: list[str], auto_top: int,
                 reps: int, aa_reps: int, timeout_s: int,
                 settle_s: float, quiet: bool = False,
                 joint: bool = False) -> dict:
    """Every arm for one page: recon, the A/A floor, then each host.

    Factored out of `main` so `src/run_live_ablation_top500.py` can drive it
    over a page sample without launching a browser per page. The A/A floor is
    measured per page rather than once for the crawl because it is a property
    of the page -- weather.com pairs to sd 122 kB, a video-heavy page will
    not -- and a cascade can only be read against its own page's floor.
    """
    res: dict = {"url": url, "reps": reps, "aa_reps": aa_reps, "hosts": {}}
    hosts = list(hosts)
    if auto_top:
        hosts += [h for h in recon(browser, url, auto_top, timeout_s=timeout_s,
                                   settle_s=settle_s) if h not in hosts]
    res["candidates"] = hosts
    if not quiet:
        print("  candidates:", ", ".join(hosts) or "(none)")

    if aa_reps:
        aa = paired(browser, url, PLACEBO_HOST, reps=aa_reps,
                    timeout_s=timeout_s, settle_s=settle_s, quiet=quiet)
        res["aa"] = {"reps": aa["reps"], "diffs": aa["diffs"],
                     "summary": summarise(aa["diffs"])}
        if not quiet:
            report("A/A", res["aa"]["summary"])

    for h in hosts:
        if not quiet:
            print(f"  ablating {h}")
        pr = paired(browser, url, h, reps=reps, timeout_s=timeout_s,
                    settle_s=settle_s, quiet=quiet)
        res["hosts"][h] = {"reps": pr["reps"], "diffs": pr["diffs"],
                           "summary": summarise(pr["diffs"]),
                           "own_mean": st.mean(pr["own"]) if pr["own"] else 0}
        if not quiet:
            report(h, res["hosts"][h]["summary"])

    # The joint arm: every candidate removed at once, which is what ETP does.
    # Its whole purpose is to be compared with the sum of the single-host arms
    # above. If the two agree, a subtree is a subtree and single-host ablation
    # measures the quantity the constant is applied to. If the joint arm is
    # larger, the hosts hold each other up -- block one and the rest of the ad
    # stack re-fetches around it -- and no amount of single-host measurement
    # adds up to the joint counterfactual. See `joint_vs_sum`.
    if joint and len(hosts) > 1:
        if not quiet:
            print(f"  ablating all {len(hosts)} at once")
        pj = paired(browser, url, *hosts, reps=reps, timeout_s=timeout_s,
                    settle_s=settle_s, quiet=quiet)
        res["joint"] = {"hosts": list(hosts), "reps": pj["reps"],
                        "diffs": pj["diffs"],
                        "summary": summarise(pj["diffs"]),
                        "own_mean": st.mean(pj["own"]) if pj["own"] else 0}
        if not quiet:
            report("JOINT", res["joint"]["summary"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--host", action="append", default=[],
                    help="tracker host to ablate; repeatable")
    ap.add_argument("--auto-top", type=int, default=0,
                    help="pick this many cascading tracker hosts from a "
                         "recon load instead")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--joint", action="store_true",
                    help="also ablate every candidate at once")
    ap.add_argument("--aa", action="store_true",
                    help="also run the placebo-vs-placebo resolution floor")
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/live_ablation")
    ap.add_argument("--timeout-s", type=int, default=45)
    ap.add_argument("--settle-s", type=float, default=5.0)
    a = ap.parse_args()

    with sync_playwright() as pw:
        browser = pw.firefox.launch(
            headless=True, firefox_user_prefs=build_prefs("normal", False))
        res = measure_page(browser, a.url, hosts=a.host, auto_top=a.auto_top,
                           reps=a.reps, aa_reps=a.reps if a.aa else 0,
                           timeout_s=a.timeout_s, settle_s=a.settle_s,
                           joint=a.joint)
        browser.close()

    d = a.out / slugify(a.url)
    d.mkdir(parents=True, exist_ok=True)
    (d / "live_cascade.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {d / 'live_cascade.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
