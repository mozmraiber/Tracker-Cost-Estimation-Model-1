"""
Parallel headless-Firefox crawl driven by a URL list. Produces HAR per page
(with `_transferSize` matching HTTP Archive's `_bytesIn` definition the
model trains against) and a per-request type log (Firefox's own
ContentPolicyType, captured via Playwright's request.resource_type).

Each worker process owns its own Firefox browser. Each page gets a fresh
BrowserContext so cookies do not carry across pages. Per-page protocol
matches src/firefox_crawl.py exactly so the 500-page result is comparable
to the 35-page sanity-check baseline:

  - Mobile emulation: 360x640, mobile UA, touch enabled, scale 2.0
  - 30-second navigation timeout
  - 5-second post-load hold for late-firing trackers
  - HAR with content omitted (we only need _transferSize)

Failure modes are categorized so the paper can report exclusion counts
honestly. The "consent_wall" classification is *not* applied here (it
needs the tracker-request count, which the analysis stage computes);
the crawler reports raw outcomes and lets analysis filter.

Output:
  <out>/har_NNNN_<slug>.json      one HAR per page
  <out>/types_NNNN_<slug>.json    URL → resource_type log per page
  <out>/_crawl_summary.json       per-page outcome metadata
"""

from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# Match src/firefox_crawl.py exactly. Do not adjust without re-running the
# 35-page baseline; any divergence breaks the comparability claim.
VIEWPORT = {"width": 360, "height": 640}
USER_AGENT = ("Mozilla/5.0 (Android 10; Mobile; rv:120.0) "
              "Gecko/120.0 Firefox/120.0")
DEVICE_SCALE = 2.0
PAGE_TIMEOUT_MS = 30_000
POST_LOAD_HOLD_MS = 5_000

# Substrings that mark a navigation_error rather than a generic failure.
# Firefox's network stack uses NS_ERROR_* codes; Playwright surfaces them
# verbatim in exception messages.
NAV_ERROR_MARKERS = (
    "NS_ERROR_UNKNOWN_HOST", "NS_ERROR_CONNECTION_REFUSED",
    "NS_ERROR_NET_TIMEOUT", "NS_ERROR_NET_RESET",
    "NS_ERROR_UNKNOWN_PROXY_HOST", "NS_ERROR_PROXY_CONNECTION_REFUSED",
    "ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION_REFUSED", "ENOTFOUND",
    "SSL", "TLS", "certificate", "SEC_ERROR",
)

_WORKER: dict = {}


def _start_browser():
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=True)
    _WORKER["pw"] = pw
    _WORKER["browser"] = browser


def _ensure_browser():
    """Return a live browser, restarting from scratch if the previous one died."""
    browser = _WORKER.get("browser")
    try:
        if browser is not None and browser.is_connected():
            return browser
    except Exception:
        pass
    # Tear down stale Playwright and start fresh
    old_pw = _WORKER.get("pw")
    if old_pw is not None:
        try:
            old_pw.stop()
        except Exception:
            pass
    _start_browser()
    return _WORKER["browser"]


def _init_worker():
    """Spin up Playwright + Firefox once per worker process. The browser
    is reused across pages within this worker (BrowserContext is fresh per
    page so there is no cookie carryover)."""
    _start_browser()
    _WORKER["pid"] = os.getpid()


def _slugify(url: str) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).rstrip("/")
    return base.replace("/", "_").replace(":", "_")[:80] or "noslug"


def _classify_error(msg: str) -> str:
    if any(m in msg for m in NAV_ERROR_MARKERS):
        return "navigation_error"
    return "other"


def crawl_page(args_tuple):
    """Worker entry point. Crawl one URL, write HAR + type log, return
    a structured per-page result. Wraps everything in try/except so that
    one bad page never kills the pool, and detects a dead browser before
    each page so a worker recovers from a Firefox crash."""
    idx, url, out_dir = args_tuple
    out_dir = Path(out_dir)
    slug = _slugify(url)
    har_path = out_dir / f"har_{idx:04d}_{slug}.json"
    type_log_path = out_dir / f"types_{idx:04d}_{slug}.json"

    info = {
        "idx": idx,
        "url": url,
        "har_path": str(har_path),
        "outcome": "unknown",
        "ok": False,
        "n_requests": 0,
        "elapsed_s": 0.0,
        "worker_pid": _WORKER.get("pid"),
        "error": "",
    }

    # Resume: skip if a non-empty HAR already exists for this slot.
    if har_path.exists() and har_path.stat().st_size > 0:
        try:
            with open(har_path) as f:
                har = json.load(f)
            info["n_requests"] = len(har.get("log", {}).get("entries", []))
            info["outcome"] = "ok_resumed"
            info["ok"] = True
            return info
        except Exception:
            # corrupt; will re-crawl
            try:
                har_path.unlink()
            except Exception:
                pass

    type_log: list[dict] = []
    t0 = time.time()
    context = None
    try:
        browser = _ensure_browser()
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
        page.on("request", on_request)

        try:
            page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
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

        try:
            page.close()
        except Exception:
            pass
    except Exception as e:
        # Catch-all so the pool never dies. Includes browser-level failures
        # (TargetClosedError, browser-died-mid-context, etc.).
        msg = f"{type(e).__name__}: {e}"
        info["outcome"] = _classify_error(msg) if not info["outcome"] in ("ok",) else info["outcome"]
        if info["outcome"] in ("unknown",):
            info["outcome"] = "browser_crash"
        info["error"] = info["error"] or msg[:200]
        # Mark browser dead so next page restarts it.
        try:
            _WORKER["browser"] = None
        except Exception:
            pass
    finally:
        if context is not None:
            try:
                context.close()  # finalizes HAR
            except Exception as e:
                info["error"] = info["error"] or f"context_close: {str(e)[:120]}"

        try:
            with open(type_log_path, "w") as f:
                json.dump(type_log, f)
        except Exception:
            pass

        info["elapsed_s"] = round(time.time() - t0, 1)

        if har_path.exists():
            try:
                with open(har_path) as f:
                    har = json.load(f)
                info["n_requests"] = len(har.get("log", {}).get("entries", []))
            except Exception:
                pass

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="Text file: one URL per line.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel browser processes.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.urls) as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    print(f"Loaded {len(urls)} URLs from {args.urls}")
    print(f"Spawning {args.workers} parallel Firefox workers...")
    print(f"Output: {out_dir}")
    print()

    payload = [(i + 1, url, str(out_dir)) for i, url in enumerate(urls)]
    summary: list[dict] = []
    t_start = time.time()

    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for info in pool.imap_unordered(crawl_page, payload, chunksize=1):
            done = len(summary) + 1
            print(f"[{done:4d}/{len(urls)}]  {info['outcome']:>20s}  "
                  f"reqs={info['n_requests']:>4d}  "
                  f"t={info['elapsed_s']:>5.1f}s  "
                  f"pid={info.get('worker_pid')}  {info['url']}",
                  flush=True)
            summary.append(info)

    elapsed = time.time() - t_start
    by_outcome: dict[str, int] = {}
    for r in summary:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1

    summary_path = out_dir / "_crawl_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_urls": len(urls),
            "n_workers": args.workers,
            "wall_clock_s": round(elapsed, 1),
            "by_outcome": by_outcome,
            "results": summary,
        }, f, indent=2)

    n_ok = sum(1 for r in summary if r["ok"])
    total_req = sum(r["n_requests"] for r in summary)

    print()
    print("=== Crawl summary ===")
    print(f"Pages attempted:  {len(summary)}")
    print(f"Pages succeeded:  {n_ok}")
    print(f"Total requests:   {total_req:,}")
    print(f"Wall clock:       {elapsed/60:.1f} min  "
          f"({elapsed/max(len(summary), 1):.1f} s/page avg)")
    print("Outcomes by category:")
    for cat, cnt in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {cat:>22s}: {cnt}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
