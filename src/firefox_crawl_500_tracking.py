"""
Paired Firefox crawl measuring what tracker blocking actually saves: the same
page loaded once with tracking-content blocking off (control) and once with it
on (as a Private Window behaves), accounting for transferred bytes and CPU on
both sides.

Per page and per arm we record:
  - every request URL, its Firefox resource_type, and its on-wire byte count
    (HAR `_transferSize`, the same definition as HTTP Archive's `_bytesIn`
    that the model trains against)
  - every request Firefox refused to make, with the nsresult that stopped it
  - CPU time burned by the whole Firefox process tree while the page loaded

WHAT THE TWO ARMS ARE
---------------------
Real Firefox Standard ETP blocks tracking *content* in Private Windows only;
in a normal window it blocks cryptominers, fingerprinters and tracking
cookies, but lets tracker requests through. Playwright cannot open a real
Private Window, so the blocking arm is assembled from the prefs a Private
Window flips:

  --mode normal    trackingprotection.enabled = false
                   trackingprotection.cryptomining.enabled = false
                   trackingprotection.fingerprinting.enabled = false
  --mode private   all three true

The control arm deliberately does *not* mirror a real normal window here: it
is a no-blocking baseline, so cryptomining and fingerprinting blocking are off
there too and every request the page made has an observed `_transferSize`.
That invariant is what the whole analysis rests on -- a request the blocking
arm refused can only be priced if the control arm actually fetched it -- and
with those protections left on, the ~20 requests a normal window blocks by
itself were missing from the baseline and unpriceable in either arm.

Everything else -- the classifier tables, the entity allowlists,
cookieBehavior=5 -- is identical across arms and pinned to Firefox's shipped
defaults (see SHARED_PREFS). The measured delta is therefore all of ETP
Standard's content blocking, not tracking content alone; each block carries
the nsresult that caused it, so compare_tracking_arms.py splits the tracking
protections from cryptomining/fingerprinting and reports them separately.
`--strict` promotes the blocking arm to ETP Strict by adding
content-track-digest256 and the social/email trackers.

Note that the request-count delta is larger than the blocked-request count,
because a blocked tracker also never injects its own subresources. Both
numbers are recorded; do not conflate them.

REPEATED VISITS
---------------
A single load of a page is a noisy measurement of it: ad auctions fill
differently on every impression, lazy content varies, and CPU carries whatever
else the machine was doing. `--repeats N` loads the whole URL list N times so
the analysis can average a page over visits and see its spread.

The rounds are sequenced to keep repeat visits of one site far apart in time.
Each round is a full pass over every URL and a round only starts once the
previous one has finished, so consecutive visits to a site are separated by an
entire pass -- hours, for a 500-URL list -- rather than by seconds. The URL
order is held identical across rounds on purpose: any reshuffle moves some site
towards the end of one round and the start of the next, which is exactly the
back-to-back pair we are trying to avoid. `--round-gap-s` adds a wall-clock
floor on top, which is what makes short lists (`--limit`) behave sensibly.

Visits are not independent of each other at the site's end -- it may recognise
a returning IP, and frequency caps shape which ads it serves -- so treat the
spread across rounds as the measurement's repeatability, not as N independent
draws.

THE BROWSER MUST BE PATCHED FIRST
---------------------------------
Playwright's Firefox ships with Remote Settings disabled in packaged JS, and
the ETP tracker lists are fetched exclusively through it, so ETP silently
blocks *nothing* out of the box -- the blocking arm would be an expensive
duplicate of the control arm. Run this once:

    python src/patch_playwright_firefox.py --apply

This crawler refuses to run a blocking arm it cannot prove is live: each
browser is warmed until the lists have synced and a canary page confirms real
blocks, and `--verify-etp` runs that check on its own.

CPU ACCOUNTING AND ITS LIMITS
-----------------------------
Firefox spreads page work across a parent process and content processes that
come and go, so a before/after read of one pid is useless. TreeCPUSampler
polls the whole process tree and tracks each pid's cumulative CPU, summing
per-pid deltas; a process that both starts and exits inside one poll interval
is missed, which in practice costs little since the processes that matter live
for the whole page.

The pool runs several browsers at once, which is much faster but means CPU
figures carry memory-bandwidth and cache contention. CPU *time* is additive so
the numbers stay meaningful, but they are inflated relative to an idle
machine, unevenly across pages. `host_cpu_pct_mean` and `n_workers` are
recorded per page so the analysis can control for it; for a headline CPU
number, re-run with `--workers 1`.

Output, per arm, under <out>/:
    har_NNNN_<slug>.json      one HAR per page
    types_NNNN_<slug>.json    URL -> resource_type log
    blocked_NNNN_<slug>.json  requests Firefox refused, with nsresult
    meta_NNNN_<slug>.json     per-page record; also the resume marker
    _crawl_summary.json       run-level outcome counts
    _pages.csv                flat per-page table for analysis

With `--repeats N` (N > 1) each round gets that same layout of its own in
<out>/repNN/, so every downstream tool can be pointed at one round unchanged,
and <out>/_pages.csv and <out>/_crawl_summary.json pool all rounds with a
`rep` column. A plain `--repeats 1` run writes the flat layout directly under
<out>/ as it always has, which is also what makes an interrupted single-round
crawl resume.

Usage:
    python src/patch_playwright_firefox.py --apply
    python src/firefox_crawl_500_tracking.py --verify-etp

    python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
        --mode normal  --out data/raw/firefox_crawl_500_tracking/normal
    # ...later, as a separate pass:
    python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
        --mode private --out data/raw/firefox_crawl_500_tracking/private

    # ten visits per page, one full pass between repeat visits of a site:
    python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
        --repeats 10 \
        --mode normal --out data/raw/firefox_crawl_500_tracking_x10/normal
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import psutil
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

# --------------------------------------------------------------------------
# Page protocol. Identical to src/firefox_crawl_500.py so that byte counts
# stay comparable with the earlier crawls; do not change one without the other.
# --------------------------------------------------------------------------
VIEWPORT = {"width": 360, "height": 640}
USER_AGENT = ("Mozilla/5.0 (Android 10; Mobile; rv:120.0) "
              "Gecko/120.0 Firefox/120.0")
DEVICE_SCALE = 2.0
PAGE_TIMEOUT_MS = 30_000
POST_LOAD_HOLD_MS = 5_000

# --------------------------------------------------------------------------
# Prefs
# --------------------------------------------------------------------------
# Applied to both arms. The first block undoes the Remote Settings lockdown in
# playwright.cfg; without it (and the omni.ja patch) the tracker lists never
# arrive. The classifier tables are Firefox's shipped defaults, written out
# explicitly so this file documents the exact list set that was in force and
# so a change in Playwright's defaults cannot silently move the arms.
SHARED_PREFS: dict = {
    "services.settings.server":
        "https://firefox.settings.services.mozilla.com/v1",
    "browser.safebrowsing.provider.mozilla.updateURL":
        "moz-sbrs:://antitracking",
    # Timestamps are char prefs, not int; an int here fails the launch.
    "browser.safebrowsing.provider.mozilla.nextupdatetime": "1",
    "browser.safebrowsing.provider.mozilla.lastupdatetime": "1",

    "urlclassifier.trackingTable":
        "moztest-track-simple,ads-track-digest256,social-track-digest256,"
        "analytics-track-digest256",
    "urlclassifier.trackingWhitelistTable":
        "moztest-trackwhite-simple,mozstd-trackwhite-digest256,"
        "google-trackwhite-digest256",
    "urlclassifier.trackingAnnotationTable":
        "moztest-track-simple,ads-track-digest256,social-track-digest256,"
        "analytics-track-digest256,content-track-digest256",
    "urlclassifier.trackingAnnotationWhitelistTable":
        "moztest-trackwhite-simple,mozstd-trackwhite-digest256,"
        "google-trackwhite-digest256",

    "network.cookie.cookieBehavior": 5,
}

# Per-arm prefs. Cryptomining and fingerprinting blocking live here, not in
# SHARED_PREFS: a real normal window applies them, but the control arm is a
# no-blocking baseline whose job is to observe the true cost of every request,
# including the ones a normal window would have refused.
ARM_PREFS = (
    "privacy.trackingprotection.enabled",
    "privacy.trackingprotection.pbmode.enabled",
    "privacy.trackingprotection.cryptomining.enabled",
    "privacy.trackingprotection.fingerprinting.enabled",
)

# ETP Strict additions, applied to the blocking arm only under --strict.
STRICT_PREFS: dict = {
    "browser.contentblocking.category": "strict",
    "privacy.annotate_channels.strict_list.enabled": True,
    "privacy.trackingprotection.socialtracking.enabled": True,
    "privacy.trackingprotection.emailtracking.enabled": True,
    "urlclassifier.trackingTable":
        "moztest-track-simple,ads-track-digest256,social-track-digest256,"
        "analytics-track-digest256,content-track-digest256",
}


def build_prefs(mode: str, strict: bool) -> dict:
    prefs = dict(SHARED_PREFS)
    for pref in ARM_PREFS:
        prefs[pref] = (mode == "private")
    if mode == "private" and strict:
        prefs.update(STRICT_PREFS)
    return prefs


# nsresults Firefox uses when the URL classifier stops a request. Anything
# matching these is a block by us, not a network failure by the site.
BLOCK_MARKERS = (
    "NS_ERROR_TRACKING_URI",
    "NS_ERROR_SOCIALTRACKING_URI",
    "NS_ERROR_CRYPTOMINING_URI",
    "NS_ERROR_FINGERPRINTING_URI",
    "NS_ERROR_EMAILTRACKING_URI",
    "NS_ERROR_MALWARE_URI",
    "NS_ERROR_UNWANTED_URI",
    "NS_ERROR_HARMFUL_URI",
    "NS_ERROR_BLOCKED_URI",
)

NAV_ERROR_MARKERS = (
    "NS_ERROR_UNKNOWN_HOST", "NS_ERROR_CONNECTION_REFUSED",
    "NS_ERROR_NET_TIMEOUT", "NS_ERROR_NET_RESET",
    "NS_ERROR_UNKNOWN_PROXY_HOST", "NS_ERROR_PROXY_CONNECTION_REFUSED",
    "ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION_REFUSED", "ENOTFOUND",
    "SSL", "TLS", "certificate", "SEC_ERROR",
)

# Canary trackers used to prove the blocking arm is live. These sit on the
# base ads/social/analytics lists and are not on the entity allowlists, so a
# warm classifier blocks them; Google properties are deliberately absent
# because google-trackwhite-digest256 exempts them.
CANARY_TRACKERS = (
    "https://static.criteo.net/js/ld/ld.js",
    "https://cdn.taboola.com/libtrc/unip/1/tfa.js",
    "https://connect.facebook.net/en_US/fbevents.js",
)
CANARY_HTML = ("<html><body>" + "".join(
    f'<script src="{u}"></script>' for u in CANARY_TRACKERS) + "</body></html>")
CANARY_ORIGIN = "https://example.com/"

_WORKER: dict = {}


# --------------------------------------------------------------------------
# CPU accounting
# --------------------------------------------------------------------------
class TreeCPUSampler:
    """Cumulative CPU time of a process tree, sampled in a background thread.

    Tracks each pid's first and last observed CPU counters and sums the
    per-pid differences, so a content process that exits mid-page still
    contributes the work it did. Processes are only ever added, never removed,
    which is what makes exited children countable.
    """

    def __init__(self, root_pid: int, interval_s: float = 0.2):
        self.root_pid = root_pid
        self.interval_s = interval_s
        self._first: dict[int, float] = {}
        self._last: dict[int, float] = {}
        self._host_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_samples = 0
        self.max_gap_s = 0.0

    def _walk(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
            procs = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        for p in procs:
            try:
                t = p.cpu_times()
                total = t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
            self._first.setdefault(p.pid, total)
            self._last[p.pid] = total

    def _loop(self) -> None:
        last_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            self.max_gap_s = max(self.max_gap_s, now - last_t)
            last_t = now
            self._walk()
            self.n_samples += 1
            try:
                self._host_samples.append(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the host-wide counter
        self._walk()                       # baseline before the page exists
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._walk()  # final read, so late CPU is not lost
        cpu = sum(self._last[pid] - self._first.get(pid, 0.0)
                  for pid in self._last)
        host = self._host_samples
        return {
            "cpu_total_s": round(cpu, 3),
            "cpu_n_procs": len(self._last),
            "cpu_n_samples": self.n_samples,
            "cpu_max_sample_gap_s": round(self.max_gap_s, 3),
            "host_cpu_pct_mean": round(sum(host) / len(host), 1) if host else None,
        }


def _firefox_root_pid() -> int | None:
    """Pid of this worker's Firefox parent process.

    Each worker owns one browser, so the search is scoped to this process's
    descendants. We deliberately exclude Playwright's Node driver: it relays
    HAR payloads, so its CPU scales with page traffic and would be double
    counted as page cost.
    """
    try:
        me = psutil.Process(os.getpid())
        for p in me.children(recursive=True):
            try:
                if "firefox" in p.name().lower():
                    return p.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except psutil.NoSuchProcess:
        pass
    return None


# --------------------------------------------------------------------------
# Browser lifecycle
# --------------------------------------------------------------------------
def _canary_blocks(browser) -> int:
    """Load a synthetic first-party page embedding known trackers.

    Returns how many of CANARY_TRACKERS the classifier refused. The page body
    is fulfilled locally so the result depends only on the classifier, not on
    some third party's markup staying put.
    """
    ctx = browser.new_context()
    try:
        page = ctx.new_page()
        blocked: set[str] = set()

        def on_failed(req):
            if any(m in (req.failure or "") for m in BLOCK_MARKERS):
                blocked.add(req.url.split("?")[0])

        page.on("requestfailed", on_failed)
        page.route(CANARY_ORIGIN, lambda route: route.fulfill(
            status=200, content_type="text/html", body=CANARY_HTML))
        page.goto(CANARY_ORIGIN, wait_until="domcontentloaded",
                  timeout=PAGE_TIMEOUT_MS)
        page.wait_for_timeout(3_000)
        return sum(1 for u in CANARY_TRACKERS if u.split("?")[0] in blocked)
    finally:
        try:
            ctx.close()
        except Exception:
            pass


def _warm_browser(browser, mode: str, warm_timeout_s: int,
                  verbose: bool = False) -> dict:
    """Wait for the classifier lists to sync before any page is measured.

    The lists arrive over the network seconds to minutes after launch. Crawling
    through that window would under-block the early pages and quietly bias the
    arm, so the blocking arm polls a canary until it sees real blocks. The
    control arm waits the same way, for symmetry of startup work, but has
    nothing to assert.
    """
    t0 = time.time()
    if mode != "private":
        # The control arm shares these prefs, so it downloads the same lists
        # at startup; it just cannot assert on them, because by construction it
        # blocks no tracking content. Wait out a matched startup allowance
        # rather than racing page one against that background work.
        time.sleep(_WORKER.get("cfg", {}).get("control_warm_s", 60))
        return {"warm_s": round(time.time() - t0, 1), "canary_blocked": 0,
                "verified": True}

    while time.time() - t0 < warm_timeout_s:
        n = _canary_blocks(browser)
        if verbose:
            print(f"    warm t+{time.time()-t0:5.1f}s  "
                  f"canary blocked {n}/{len(CANARY_TRACKERS)}", flush=True)
        if n == len(CANARY_TRACKERS):
            return {"warm_s": round(time.time() - t0, 1),
                    "canary_blocked": n, "verified": True}
        time.sleep(5)

    n = _canary_blocks(browser)
    return {"warm_s": round(time.time() - t0, 1), "canary_blocked": n,
            "verified": n > 0}


def _start_browser():
    cfg = _WORKER["cfg"]
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=True,
                                firefox_user_prefs=cfg["prefs"])
    _WORKER["pw"] = pw
    _WORKER["browser"] = browser
    _WORKER["ff_pid"] = _firefox_root_pid()

    warm = _warm_browser(browser, cfg["mode"], cfg["warm_timeout_s"])
    _WORKER["warm"] = warm
    if cfg["mode"] == "private" and not warm["verified"]:
        # Refuse to produce a blocking arm that does not block: that failure
        # is invisible in the output but wrecks every downstream number.
        raise RuntimeError(
            "ETP did not block the canary trackers within "
            f"{cfg['warm_timeout_s']}s. Run "
            "`python src/patch_playwright_firefox.py --check` -- an unpatched "
            "Playwright Firefox cannot sync the tracker lists."
        )
    return browser


def _ensure_browser():
    browser = _WORKER.get("browser")
    try:
        if browser is not None and browser.is_connected():
            return browser
    except Exception:
        pass
    old_pw = _WORKER.get("pw")
    if old_pw is not None:
        try:
            old_pw.stop()
        except Exception:
            pass
    return _start_browser()


def _init_worker(cfg: dict):
    _WORKER["cfg"] = cfg
    _WORKER["pid"] = os.getpid()
    _start_browser()


# --------------------------------------------------------------------------
# Per-page crawl
# --------------------------------------------------------------------------
def _slugify(url: str) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).rstrip("/")
    return base.replace("/", "_").replace(":", "_")[:80] or "noslug"


def _classify_error(msg: str) -> str:
    if any(m in msg for m in NAV_ERROR_MARKERS):
        return "navigation_error"
    return "other"


def _rollup_har(har_path: Path, type_log: list[dict]) -> dict:
    """Per-page byte accounting from the HAR.

    `_transferSize` is on-wire bytes, matching HTTP Archive's `_bytesIn`. A
    revalidated or cached entry reports -1 or 0; those are counted as zero
    bytes but still counted as requests, and reported separately so the
    analysis can see how much of a page was not actually transferred.
    """
    out = {
        "n_requests": 0,
        "transfer_bytes": 0,
        "n_no_transfer_size": 0,
        "bytes_by_type": {},
        "n_by_type": {},
    }
    if not har_path.exists():
        return out
    try:
        with open(har_path) as f:
            har = json.load(f)
    except Exception:
        return out

    types = {}
    for e in type_log:
        types.setdefault(e["url"], e.get("resource_type") or "other")

    for entry in har.get("log", {}).get("entries", []):
        url = entry.get("request", {}).get("url", "")
        size = entry.get("response", {}).get("_transferSize")
        if size is None or size < 0:
            size = 0
            out["n_no_transfer_size"] += 1
        rtype = types.get(url, "other")
        out["n_requests"] += 1
        out["transfer_bytes"] += size
        out["bytes_by_type"][rtype] = out["bytes_by_type"].get(rtype, 0) + size
        out["n_by_type"][rtype] = out["n_by_type"].get(rtype, 0) + 1
    return out


def crawl_page(args_tuple):
    """Worker entry point: crawl one URL in one arm.

    Everything is wrapped so a single bad page can never kill the pool, and a
    dead browser is detected before each page so a worker survives a Firefox
    crash.
    """
    idx, url, out_dir, mode, rep = args_tuple
    out_dir = Path(out_dir)
    slug = _slugify(url)
    har_path = out_dir / f"har_{idx:04d}_{slug}.json"
    type_log_path = out_dir / f"types_{idx:04d}_{slug}.json"
    blocked_path = out_dir / f"blocked_{idx:04d}_{slug}.json"
    meta_path = out_dir / f"meta_{idx:04d}_{slug}.json"

    # Resume on the meta file rather than the HAR: a HAR alone has no CPU
    # measurement attached, and resuming from one would leave holes.
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                prev = json.load(f)
            prev["outcome"] = prev.get("outcome", "ok") + "_resumed"
            return prev
        except Exception:
            pass

    info = {
        "idx": idx,
        "url": url,
        "rep": rep,
        "mode": mode,
        "outcome": "unknown",
        "ok": False,
        "t_start_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_requests": 0,
        "transfer_bytes": 0,
        "n_blocked": 0,
        "blocked_bytes_unknown": True,
        "elapsed_s": 0.0,
        "worker_pid": _WORKER.get("pid"),
        "n_workers": _WORKER.get("cfg", {}).get("n_workers"),
        "error": "",
    }

    type_log: list[dict] = []
    blocked: list[dict] = []
    all_failed: list[dict] = []
    t0 = time.time()
    context = None
    sampler = None
    try:
        browser = _ensure_browser()
        ff_pid = _WORKER.get("ff_pid") or _firefox_root_pid()
        _WORKER["ff_pid"] = ff_pid

        # Start sampling before the context exists so content processes spawned
        # for this page are counted from birth.
        if ff_pid is not None:
            sampler = TreeCPUSampler(ff_pid)
            sampler.start()

        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            has_touch=True,
            is_mobile=False,
            device_scale_factor=DEVICE_SCALE,
            record_har_path=str(har_path),
            record_har_content="omit",
        )
        page = context.new_page()

        def on_request(req):
            type_log.append({
                "url": req.url,
                "resource_type": req.resource_type,
                "method": req.method,
            })

        def on_failed(req):
            failure = req.failure or ""
            rec = {
                "url": req.url,
                "resource_type": req.resource_type,
                "failure": failure,
            }
            all_failed.append(rec)
            if any(m in failure for m in BLOCK_MARKERS):
                blocked.append(rec)

        page.on("request", on_request)
        page.on("requestfailed", on_failed)

        try:
            page.goto(url, timeout=PAGE_TIMEOUT_MS,
                      wait_until="domcontentloaded")
            page.wait_for_timeout(POST_LOAD_HOLD_MS)
            info["ok"] = True
            info["outcome"] = "ok"
        except PWTimeout as e:
            info["outcome"] = "timeout"
            info["error"] = str(e)[:200]
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            info["outcome"] = _classify_error(msg)
            info["error"] = msg[:200]

        # Stop before teardown: context.close() flushes the HAR, and that
        # cost is measurement overhead rather than page cost.
        if sampler is not None:
            info.update(sampler.stop())
            sampler = None

        try:
            page.close()
        except Exception:
            pass
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if info["outcome"] == "unknown":
            info["outcome"] = "browser_crash"
        info["error"] = info["error"] or msg[:200]
        _WORKER["browser"] = None  # force a fresh browser for the next page
    finally:
        if sampler is not None:
            info.update(sampler.stop())
        if context is not None:
            try:
                context.close()  # finalizes the HAR
            except Exception as e:
                info["error"] = info["error"] or f"context_close: {str(e)[:120]}"

        for path, payload in ((type_log_path, type_log),
                              (blocked_path, {"blocked": blocked,
                                              "all_failed": all_failed})):
            try:
                with open(path, "w") as f:
                    json.dump(payload, f)
            except Exception:
                pass

        info["elapsed_s"] = round(time.time() - t0, 1)
        info["n_blocked"] = len(blocked)
        info["n_failed_other"] = len(all_failed) - len(blocked)
        info.update(_rollup_har(har_path, type_log))
        info["warm_s"] = (_WORKER.get("warm") or {}).get("warm_s")
        info["canary_blocked"] = (_WORKER.get("warm") or {}).get("canary_blocked")

        try:
            with open(meta_path, "w") as f:
                json.dump(info, f, indent=2)
        except Exception:
            pass

    return info


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
CSV_COLUMNS = [
    "idx", "url", "rep", "mode", "outcome", "ok", "t_start_iso",
    "n_requests", "transfer_bytes", "n_no_transfer_size",
    "n_blocked", "n_failed_other",
    "cpu_total_s", "cpu_n_procs", "cpu_n_samples", "cpu_max_sample_gap_s",
    "host_cpu_pct_mean", "n_workers", "elapsed_s", "warm_s", "canary_blocked",
    "worker_pid", "error",
]


def round_dir(out_dir: Path, rep: int, n_repeats: int) -> Path:
    """Where one round writes.

    A single-round crawl keeps the flat historical layout directly under
    <out>, so existing output directories still resume and every downstream
    tool that globs an arm directory is unaffected. Repeats get one such
    directory each, which keeps those tools usable a round at a time.
    """
    return out_dir if n_repeats == 1 else out_dir / f"rep{rep:02d}"


def write_pages_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows,
                        key=lambda r: (r.get("rep", 1), r.get("idx", 0))):
            w.writerow(r)


def write_arm_outputs(dest: Path, rows: list[dict], *, args, prefs: dict,
                      n_urls: int, elapsed_s: float, extra: dict) -> dict:
    """Write `_crawl_summary.json` and `_pages.csv` for a round, or for the
    whole run. Returns the outcome counts so the caller can report them."""
    by_outcome: dict[str, int] = {}
    for r in rows:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    cpus = [r["cpu_total_s"] for r in rows if r.get("cpu_total_s")]

    payload = {
        "mode": args.mode,
        "strict": args.strict,
        "prefs": prefs,
        "n_urls": n_urls,
        "n_workers": args.workers,
        "urls_file": args.urls,
        "finished_iso": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "wall_clock_s": round(elapsed_s, 1),
        "by_outcome": by_outcome,
        "totals": {
            "transfer_bytes": sum(r["transfer_bytes"] for r in rows),
            "n_requests": sum(r["n_requests"] for r in rows),
            "n_blocked": sum(r["n_blocked"] for r in rows),
            "cpu_total_s": round(sum(cpus), 1) if cpus else None,
        },
        "results": rows,
    }
    payload.update(extra)
    with open(dest / "_crawl_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_pages_csv(dest / "_pages.csv", rows)
    return by_outcome


def cmd_verify_etp(strict: bool, warm_timeout_s: int) -> int:
    """Prove the blocking arm can block before spending hours on a crawl."""
    print("Launching Firefox with the blocking arm's prefs...")
    prefs = build_prefs("private", strict)
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True, firefox_user_prefs=prefs)
        print(f"Firefox {browser.version}")
        print(f"Warming the classifier (up to {warm_timeout_s}s); the tracker "
              "lists sync over the network after launch.")
        warm = _warm_browser(browser, "private", warm_timeout_s, verbose=True)
        browser.close()

    n, total = warm["canary_blocked"], len(CANARY_TRACKERS)
    print()
    if n == total:
        print(f"PASS: all {total} canary trackers blocked after "
              f"{warm['warm_s']}s. ETP is live.")
        return 0
    if n > 0:
        print(f"PARTIAL: {n}/{total} canary trackers blocked after "
              f"{warm['warm_s']}s.")
        print("ETP is working but the lists may still be syncing; consider a "
              "longer --warm-timeout.")
        return 0
    print(f"FAIL: 0/{total} canary trackers blocked. ETP is not blocking.")
    print("Check the browser patch:")
    print("  python src/patch_playwright_firefox.py --check")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", help="Text file: one URL per line.")
    ap.add_argument("--out", help="Output directory for this arm.")
    ap.add_argument("--mode", choices=("normal", "private"),
                    help="normal = tracking content allowed (control); "
                         "private = blocked, as a Private Window behaves.")
    ap.add_argument("--strict", action="store_true",
                    help="Blocking arm uses ETP Strict rather than Standard.")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel browsers. Use 1 for clean CPU numbers.")
    ap.add_argument("--limit", type=int,
                    help="Crawl only the first N URLs (for a smoke test).")
    ap.add_argument("--repeats", type=int, default=1, metavar="N",
                    help="Visit every URL N times. Each round is a full pass "
                         "over the list and waits for the previous one to "
                         "finish, so repeat visits of a site are a whole pass "
                         "apart. N>1 writes each round to <out>/repNN/.")
    ap.add_argument("--round-gap-s", type=float, default=0.0, metavar="S",
                    help="Floor on the wall-clock period between the starts "
                         "of consecutive rounds. A full pass usually exceeds "
                         "it on its own; it is what keeps repeat visits apart "
                         "on a short list.")
    ap.add_argument("--warm-timeout", type=int, default=240,
                    help="Max seconds to wait for the tracker lists to sync "
                         "in the blocking arm (polls a canary, exits early).")
    ap.add_argument("--control-warm-s", type=int, default=60,
                    help="Matched startup allowance for the control arm, "
                         "which downloads the same lists but cannot assert "
                         "on them.")
    ap.add_argument("--verify-etp", action="store_true",
                    help="Check that ETP blocks, then exit.")
    args = ap.parse_args()

    if args.verify_etp:
        return cmd_verify_etp(args.strict, args.warm_timeout)

    if args.repeats < 1:
        ap.error("--repeats must be at least 1")

    missing = [f"--{n}" for n in ("urls", "out", "mode")
               if getattr(args, n) is None]
    if missing:
        ap.error("required unless --verify-etp: " + ", ".join(missing))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.urls) as f:
        urls = [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]
    if args.limit:
        urls = urls[:args.limit]

    prefs = build_prefs(args.mode, args.strict)
    cfg = {
        "mode": args.mode,
        "prefs": prefs,
        "warm_timeout_s": args.warm_timeout,
        "control_warm_s": args.control_warm_s,
        "n_workers": args.workers,
    }

    etp_label = ("Strict" if args.strict else "Standard") \
        if args.mode == "private" else "off"
    n_repeats = args.repeats  # validated >= 1 above
    n_pages = len(urls) * n_repeats
    print(f"Arm:        {args.mode}  (tracking-content blocking: {etp_label})")
    print(f"URLs:       {len(urls)} from {args.urls}")
    if n_repeats > 1:
        gap = (f", at least {args.round_gap_s:.0f}s apart"
               if args.round_gap_s else "")
        print(f"Repeats:    {n_repeats} rounds{gap}  "
              f"({n_pages} page loads; a site is revisited only after a full "
              "pass over the list)")
    print(f"Workers:    {args.workers}"
          + ("" if args.workers == 1 else
             "   [CPU figures carry contention; use --workers 1 for clean CPU]"))
    print(f"Output:     {out_dir}")
    print()

    summary: list[dict] = []
    rounds: list[dict] = []
    t_start = time.time()
    t_round_start = None

    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(cfg,)) as pool:
        for rnd in range(1, n_repeats + 1):
            # The floor is on the period between round *starts*, because that
            # is what sets the gap between the two visits to a given site:
            # each keeps its position in the pass.
            if t_round_start is not None and args.round_gap_s > 0:
                wait = args.round_gap_s - (time.time() - t_round_start)
                if wait > 0:
                    print(f"--- waiting {wait:.0f}s before round {rnd} "
                          "(--round-gap-s)", flush=True)
                    time.sleep(wait)
            t_round_start = time.time()

            rdir = round_dir(out_dir, rnd, n_repeats)
            rdir.mkdir(parents=True, exist_ok=True)
            if n_repeats > 1:
                print(f"--- round {rnd}/{n_repeats} -> {rdir}", flush=True)

            # Identical URL order every round, on purpose: a reshuffle would
            # put some site late in one round and early in the next, which is
            # the back-to-back pair the sequencing exists to prevent.
            payload = [(i + 1, url, str(rdir), args.mode, rnd)
                       for i, url in enumerate(urls)]
            round_rows: list[dict] = []
            tag = f"r{rnd:02d}  " if n_repeats > 1 else ""
            for info in pool.imap_unordered(crawl_page, payload, chunksize=1):
                round_rows.append(info)
                done = len(summary) + len(round_rows)
                cpu = info.get("cpu_total_s")
                print(f"[{done:5d}/{n_pages}]  {tag}"
                      f"{info['outcome']:>18s}  "
                      f"reqs={info['n_requests']:>4d}  "
                      f"{info['transfer_bytes']/1e6:>6.2f}MB  "
                      f"blocked={info['n_blocked']:>3d}  "
                      f"cpu={cpu if cpu is None else f'{cpu:5.1f}'}s  "
                      f"t={info['elapsed_s']:>5.1f}s  {info['url']}",
                      flush=True)

            round_elapsed = time.time() - t_round_start
            if n_repeats > 1:
                write_arm_outputs(rdir, round_rows, args=args, prefs=prefs,
                                  n_urls=len(urls), elapsed_s=round_elapsed,
                                  extra={"rep": rnd, "n_repeats": n_repeats})
            rounds.append({
                "rep": rnd,
                "dir": str(rdir),
                "wall_clock_s": round(round_elapsed, 1),
                "started_iso": datetime.fromtimestamp(
                    t_round_start, timezone.utc).isoformat(
                        timespec="seconds"),
                "n_ok": sum(1 for r in round_rows if r["ok"]),
                "transfer_bytes": sum(r["transfer_bytes"]
                                      for r in round_rows),
                "n_blocked": sum(r["n_blocked"] for r in round_rows),
            })
            summary.extend(round_rows)

    elapsed = time.time() - t_start
    ok = [r for r in summary if r["ok"]]
    total_bytes = sum(r["transfer_bytes"] for r in summary)
    total_blocked = sum(r["n_blocked"] for r in summary)
    cpus = [r["cpu_total_s"] for r in summary if r.get("cpu_total_s")]

    by_outcome = write_arm_outputs(
        out_dir, summary, args=args, prefs=prefs, n_urls=len(urls),
        elapsed_s=elapsed,
        extra={"n_repeats": n_repeats,
               "round_gap_s": args.round_gap_s,
               "rounds": rounds})

    print()
    print("=== Crawl summary ===")
    print(f"Arm:              {args.mode} (blocking: {etp_label})")
    if n_repeats > 1:
        print(f"Rounds:           {n_repeats} x {len(urls)} URLs")
    print(f"Pages attempted:  {len(summary)}")
    print(f"Pages succeeded:  {len(ok)}")
    print(f"Total requests:   {sum(r['n_requests'] for r in summary):,}")
    print(f"Total transfer:   {total_bytes/1e6:,.1f} MB")
    print(f"Requests blocked: {total_blocked:,}")
    if cpus:
        print(f"CPU (sum):        {sum(cpus):,.0f} s over {len(cpus)} pages "
              f"({sum(cpus)/len(cpus):.1f} s/page mean)")
    print(f"Wall clock:       {elapsed/60:.1f} min "
          f"({elapsed/max(len(summary), 1):.1f} s/page avg)")
    print("Outcomes:")
    for cat, cnt in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {cat:>22s}: {cnt}")
    if n_repeats > 1:
        print("Rounds (a site's revisit gap is the gap between round "
              "starts):")
        prev = None
        for r in rounds:
            t = datetime.fromisoformat(r["started_iso"])
            if prev is None:
                gap = ""
            else:
                dt = (t - prev).total_seconds()
                gap = ("  +%.0f s since previous" % dt if dt < 600
                       else "  +%.0f min since previous" % (dt / 60))
            prev = t
            print(f"  r{r['rep']:02d}  {r['started_iso']}  "
                  f"{r['wall_clock_s']/60:5.1f} min  "
                  f"ok={r['n_ok']:>4d}  "
                  f"{r['transfer_bytes']/1e6:>8.1f} MB{gap}")
    print(f"Per-page table:   {out_dir / '_pages.csv'}"
          + ("   (pooled; `rep` column, per-round copies in repNN/)"
             if n_repeats > 1 else ""))

    if args.mode == "private" and total_blocked == 0:
        print()
        print("WARNING: the blocking arm blocked nothing across every page.")
        print("Treat this arm as invalid and check "
              "`python src/patch_playwright_firefox.py --check`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
