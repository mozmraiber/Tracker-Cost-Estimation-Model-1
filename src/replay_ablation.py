"""Measure a tracker's follow-up bytes by ablating it from a *replayed* page.

WHY THIS EXISTS
---------------
Every cascade instrument in this repo is loud, and all of them are loud for
the same reason: they compare two live page loads, and a live page load is
mostly first-party churn. `src/firefox_crawl_500_tracking.py` blocks every
tracker on a page and reads the whole-page delta, which carries 74% of noise
per pass and 23.4% pooled over ten; `bytes_saved_tracking` is four times
quieter only because it throws away everything the Disconnect list cannot
name, which is exactly the part of the cascade the shipped 47 KB is missing.

A replayed page has no churn. Record one real load's responses, serve them
back from disk, and the load becomes a deterministic function of what the
browser asks for. Then blocking one tracker and diffing the total is not a
noisy difference of two samples -- it is the tracker's subtree, including the
ad creatives and iframes on hosts no list names, which is the one quantity
`listed_factor_from_delta` says it cannot see.

WHAT IT COSTS
-------------
Fidelity, and the script reports it rather than assuming it. A replayed page
asks for URLs that are not in the recording -- cache busters, auction ids,
anything with a nonce -- and those are aborted and counted. `--reps` replays
the baseline several times so the residual non-determinism is visible next to
the cascade it is supposed to be smaller than. A page whose unmatched share
is large is not evidence about anything and should be dropped, not corrected.

The second cost is behavioural: a frozen ad response is a *stale* ad response,
and a script that would have run an auction may take a different branch when
its bid comes back identical every time. That is a real limit on what replay
measures and no amount of repetition fixes it; see the docstring on
`cascade_for`.

USAGE
-----
    python src/replay_ablation.py probe --url https://www.example.com \
        --out data/raw/replay_ablation --reps 3 --top 5

Writes `<out>/<slug>/page.har.zip` (the recording, bodies attached) and
`<out>/<slug>/cascade.json`, and prints the table.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from compare_estimate_vs_etp import is_third_party  # noqa: E402
from firefox_crawl_500_tracking import build_prefs  # noqa: E402

try:
    import disconnect
except ImportError:  # pragma: no cover
    disconnect = None

#: A host no recording contains, for the placebo arm. See `probe`.
PLACEBO_HOST = "placebo.invalid"

#: Resource types that can pull in a subtree. The same split
#: `FOLLOWUP_BYTES_PER_REQUEST` makes: only code cascades.
CASCADING_TYPES = ("script", "document", "xhr", "fetch")


def slugify(url: str) -> str:
    host = urlsplit(url).netloc or url
    return re.sub(r"[^a-zA-Z0-9.-]", "_", host)[:60]


def is_tracker(url: str) -> bool:
    if disconnect is None:
        return False
    try:
        return bool(disconnect.is_tracker(url))
    except Exception:
        return False


# --------------------------------------------------------------------------
# recording


def record(url: str, har_zip: Path, *, timeout_s: int, settle_s: float) -> dict:
    """One live control-arm load, with bodies, into `har_zip`.

    Control-arm prefs exactly: no tracking, cryptomining or fingerprinting
    blocking, so the recording contains every request the page makes and the
    ablation below is the only thing that ever removes one.
    """
    har_zip.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.firefox.launch(
            headless=True, firefox_user_prefs=build_prefs("normal", False))
        ctx = browser.new_context(
            record_har_path=str(har_zip),
            record_har_content="attach",
            record_har_mode="full",
        )
        page = ctx.new_page()
        status = "ok"
        try:
            page.goto(url, wait_until="load", timeout=timeout_s * 1000)
        except Exception as exc:
            status = f"goto: {type(exc).__name__}"
        page.wait_for_timeout(int(settle_s * 1000))
        ctx.close()
        browser.close()
    return {"status": status}


def har_index(har_zip: Path) -> dict:
    """(method, url) -> transfer bytes, plus the entry list, from a HAR zip.

    `_transferSize` is the same definition the rest of the repo counts in --
    HTTP Archive's `_bytesIn` -- so a replayed request is charged what it
    actually cost on the wire during the recording, not the size of the body
    the replay server hands back.
    """
    with zipfile.ZipFile(har_zip) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".har"))
        har = json.loads(zf.read(name))
    sizes: dict[tuple[str, str], int] = {}
    entries = []
    for e in har.get("log", {}).get("entries", []):
        req = e.get("request", {})
        u, m = req.get("url", ""), req.get("method", "GET")
        if not u:
            continue
        size = e.get("response", {}).get("_transferSize")
        size = 0 if size is None or size < 0 else int(size)
        sizes.setdefault((m, u), size)
        entries.append({
            "url": u, "method": m, "bytes": size,
            "type": e.get("_resourceType") or "other",
            "tracker": is_tracker(u),
            "host": urlsplit(u).netloc,
        })
    return {"sizes": sizes, "entries": entries}


# --------------------------------------------------------------------------
# replay


def replay(url: str, har_zip: Path, sizes: dict, *,
           block_hosts: frozenset = frozenset(),
           block_urls: frozenset = frozenset(),
           timeout_s: int, settle_s: float) -> dict:
    """Load `url` with every response served from `har_zip`.

    Anything in `block_hosts`/`block_urls` is aborted before the HAR router
    sees it, which is the ablation. Anything the recording does not contain is
    aborted by the router and counted separately: that is the fidelity
    number, and it is the reason this script reports a baseline replay
    alongside every ablation rather than trusting one.
    """
    served: list[tuple[str, str]] = []
    failed: list[str] = []
    ablated: list[str] = []

    def want_blocked(u: str) -> bool:
        if u in block_urls:
            return True
        return urlsplit(u).netloc in block_hosts

    with sync_playwright() as pw:
        browser = pw.firefox.launch(
            headless=True, firefox_user_prefs=build_prefs("normal", False))
        ctx = browser.new_context(service_workers="block")
        # Registration order matters: the ablation route is added last so it
        # wins over the HAR router for the URLs it claims.
        ctx.route_from_har(har_zip, not_found="abort")
        if block_hosts or block_urls:
            def _abort(route):
                ablated.append(route.request.url)
                route.abort()
            ctx.route(want_blocked, _abort)

        page = ctx.new_page()
        page.on("request", lambda r: served.append((r.method, r.url)))
        page.on("requestfailed", lambda r: failed.append(r.url))
        status = "ok"
        try:
            page.goto(url, wait_until="load", timeout=timeout_s * 1000)
        except Exception as exc:
            status = f"goto: {type(exc).__name__}"
        page.wait_for_timeout(int(settle_s * 1000))
        ctx.close()
        browser.close()

    failed_set = set(failed)
    ablated_set = set(ablated)
    total = 0
    matched = unmatched = 0
    unmatched_urls = []
    host_bytes: dict[str, int] = {}
    host_n: dict[str, int] = {}
    for m, u in served:
        if u in ablated_set:
            continue
        if u in failed_set or (m, u) not in sizes:
            unmatched += 1
            unmatched_urls.append(u)
            continue
        size = sizes[(m, u)]
        matched += 1
        total += size
        h = urlsplit(u).netloc
        host_bytes[h] = host_bytes.get(h, 0) + size
        host_n[h] = host_n.get(h, 0) + 1
    return {
        "status": status,
        "requests": len(served),
        "matched": matched,
        "unmatched": unmatched,
        "ablated": len(ablated),
        "bytes": total,
        "host_bytes": host_bytes,
        "host_n": host_n,
        "unmatched_urls": unmatched_urls[:20],
    }


def cascade_for(url: str, har_zip: Path, sizes: dict, host: str, *,
                placebo: dict, **kw) -> dict:
    """Bytes the page stops fetching when `host` is ablated, less its own.

    Own bytes are read off the *placebo replay*, not off the recording. The
    two differ: a replayed page does not ask for every URL the live one did
    (lazy content below the fold, mostly), so charging a host the bytes it
    spent live while the ablation can only remove what the replay asked for
    subtracts a cost the measurement never paid -- which showed up as hosts
    with a negative cascade before this was fixed.

    This is the subtree, not a counterfactual about a live page: the responses
    everything else receives are frozen, so nothing downstream can react to
    the block by fetching something different. Against a live blocked arm that
    cuts both ways -- a real page might fall back to a house ad, or retry --
    and replay will not see either.
    """
    r = replay(url, har_zip, sizes, block_hosts=frozenset([host]), **kw)
    own = placebo["host_bytes"].get(host, 0)
    r["host"] = host
    r["own_bytes"] = own
    r["own_requests"] = placebo["host_n"].get(host, 0)
    r["cascade_bytes"] = placebo["bytes"] - r["bytes"] - own
    # Requests that vanished but could not be priced, because the recording
    # had no response for them. They are charged zero above, so the cascade
    # is a lower bound by however many of these there are.
    r["unpriced_dropped"] = placebo["unmatched"] - r["unmatched"]
    return r


# --------------------------------------------------------------------------


def probe(url: str, out: Path, *, reps: int, top: int,
          timeout_s: int, settle_s: float, max_spread: int = 20_000) -> dict:
    d = out / slugify(url)
    har_zip = d / "page.har.zip"
    if not har_zip.exists():
        print(f"recording {url} ...", flush=True)
        record(url, har_zip, timeout_s=timeout_s, settle_s=settle_s)
    idx = har_index(har_zip)
    sizes, entries = idx["sizes"], idx["entries"]
    live_bytes = sum(e["bytes"] for e in entries)
    print(f"recorded {len(entries)} requests, {live_bytes/1e6:.2f} MB live")

    kw = dict(timeout_s=timeout_s, settle_s=settle_s)
    bases = [replay(url, har_zip, sizes, **kw) for _ in range(reps)]
    for i, b in enumerate(bases):
        print(f"  baseline rep {i}: {b['bytes']/1e6:7.3f} MB  "
              f"{b['matched']:4d} matched  {b['unmatched']:3d} unmatched")

    # The placebo arm. Registering an ablation route makes Playwright
    # intercept *every* request to evaluate the predicate, which costs time
    # the page can notice, so a run with a route is not comparable to a run
    # without one. The placebo registers the identical machinery against a
    # host the recording never contains: its delta from the baseline is the
    # instrument's own footprint, and every cascade below is measured against
    # it rather than against the baseline.
    placebos = [replay(url, har_zip, sizes,
                       block_hosts=frozenset([PLACEBO_HOST]), **kw)
                for _ in range(reps)]
    for i, b in enumerate(placebos):
        print(f"  placebo  rep {i}: {b['bytes']/1e6:7.3f} MB  "
              f"{b['matched']:4d} matched  {b['unmatched']:3d} unmatched")
    baseline_bytes = st.median([b["bytes"] for b in placebos])
    spread = (max(b["bytes"] for b in placebos)
              - min(b["bytes"] for b in placebos))
    footprint = st.median([b["bytes"] for b in bases]) - baseline_bytes
    print(f"  route footprint {footprint/1e3:.1f} kB; "
          f"placebo spread {spread/1e3:.1f} kB over {reps} reps")

    # A page whose replay is not deterministic is not a measurement, and the
    # failure is not subtle when it happens: globo.com's replay alternates
    # between two states 2.9 MB apart, which read as cascades of -3 MB for
    # every host tried. Refuse rather than report. The threshold is absolute
    # because what it has to be small against is a cascade of tens of kB, not
    # a page of several MB.
    usable = spread <= max_spread
    if not usable:
        print(f"  UNUSABLE: placebo spread {spread/1e3:.1f} kB exceeds "
              f"{max_spread/1e3:.1f} kB -- this page's replay is not "
              f"deterministic, so no cascade below it would mean anything")

    # Candidate hosts: Disconnect-matched, serving something that can cascade.
    by_host: dict[str, dict] = {}
    for e in entries:
        # Third-party as well as listed: Disconnect names the ad-tech vendors
        # themselves, so on globo.com the page's own host came up as a
        # candidate and ablating it removed the site.
        if not (e["tracker"] and is_third_party(e["url"], url)):
            continue
        h = by_host.setdefault(e["host"], {"bytes": 0, "n": 0, "casc": False})
        h["bytes"] += e["bytes"]
        h["n"] += 1
        h["casc"] |= e["type"] in CASCADING_TYPES
    cands = sorted((h for h, v in by_host.items() if v["casc"]),
                   key=lambda h: -by_host[h]["bytes"])[:top]

    # Rank by what the *replay* fetched from the host, since that is all an
    # ablation can remove. A host the replay never asked for is not a
    # measurement of zero cascade; it is not a measurement.
    ref = placebos[[b["bytes"] for b in placebos].index(baseline_bytes)]
    cands = [h for h in cands if ref["host_bytes"].get(h, 0) > 0]

    rows = []
    for h in (cands if usable else []):
        r = cascade_for(url, har_zip, sizes, h, placebo=ref, **kw)
        rows.append(r)
        print(f"  ablate {h:34.34s} own {r['own_bytes']/1e3:7.1f} kB "
              f"cascade {r['cascade_bytes']/1e3:8.1f} kB "
              f"({r['ablated']} ablated, {r['unpriced_dropped']:+d} unpriced)")

    res = {
        "url": url, "live_bytes": live_bytes,
        "baseline_bytes": baseline_bytes, "baseline_spread_bytes": spread,
        "route_footprint_bytes": footprint, "usable": usable,
        "baseline_reps": bases, "placebo_reps": placebos, "ablations": rows,
    }
    (d / "cascade.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "data/raw/replay_ablation")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--timeout-s", type=int, default=45)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--max-spread-kb", type=float, default=20.0,
                   help="refuse to report cascades if the placebo arm moves "
                        "by more than this across reps")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.url, a.out, reps=a.reps, top=a.top,
              timeout_s=a.timeout_s, settle_s=a.settle_s,
              max_spread=int(a.max_spread_kb * 1000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
