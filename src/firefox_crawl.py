"""
Drive headless Firefox via Playwright through a curated list of real publisher
pages, recording HAR for each. The HAR's `_transferSize` field is the actual
on-wire byte count (matches HTTP Archive's `_bytesIn` definition that the
model is trained against), so this gives a faithful in-page ground truth for
Claim B: model predictions vs Firefox-deployment-realistic transfer bytes.

Output: data/raw/firefox_crawl/har_*.json (one per page) and a summary CSV.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "firefox_crawl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of high-traffic publisher pages spanning categories likely to
# have heavy tracker presence. Intentionally not logged-in / not paywalled.
TEST_PAGES = [
    # News
    "https://www.cnn.com",
    "https://www.bbc.com",
    "https://www.foxnews.com",
    "https://www.theguardian.com/us",
    "https://www.washingtonpost.com",
    "https://www.reuters.com",
    "https://www.npr.org",
    "https://www.bloomberg.com",
    "https://www.dailymail.co.uk",
    "https://www.huffpost.com",
    # E-commerce
    "https://www.amazon.com",
    "https://www.ebay.com",
    "https://www.etsy.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bestbuy.com",
    # Social / Forum
    "https://www.reddit.com",
    "https://www.quora.com",
    "https://medium.com",
    "https://www.pinterest.com",
    # Tech
    "https://techcrunch.com",
    "https://www.theverge.com",
    "https://arstechnica.com",
    "https://www.wired.com",
    "https://www.engadget.com",
    "https://mashable.com",
    "https://gizmodo.com",
    # Sports / Entertainment
    "https://www.espn.com",
    "https://www.imdb.com",
    "https://www.rottentomatoes.com",
    # Lifestyle / Misc
    "https://www.buzzfeed.com",
    "https://www.vox.com",
    "https://www.vice.com",
    "https://weather.com",
    "https://www.allrecipes.com",
]


def crawl_one(page, url: str, har_path: Path, type_log_path: Path) -> dict:
    """Navigate Firefox to URL, record HAR, also log per-request resourceType
    (Firefox's own ContentPolicyType classification) since HAR doesn't preserve it."""
    info = {"url": url, "har_path": str(har_path), "ok": False, "n_requests": 0}
    type_log = []

    def on_request(req):
        type_log.append({
            "url": req.url,
            "resource_type": req.resource_type,
            "method": req.method,
        })

    page.on("request", on_request)

    try:
        page.goto(url, timeout=30_000, wait_until="domcontentloaded")
        page.wait_for_timeout(5_000)
        info["ok"] = True
    except Exception as e:
        info["error"] = str(e)[:120]

    # Persist the type log alongside the HAR
    with open(type_log_path, "w") as f:
        json.dump(type_log, f)
    return info


def main():
    summary = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)

        for i, url in enumerate(TEST_PAGES, 1):
            slug = url.replace("https://", "").replace("/", "_").rstrip("_")
            har_path = OUT_DIR / f"har_{i:02d}_{slug}.json"
            type_log_path = OUT_DIR / f"types_{i:02d}_{slug}.json"

            # Mobile emulation matching HTTP Archive's Moto G4 settings:
            # 360x640 viewport, mobile UA, touch enabled. This is what the
            # training labels were collected under, and matches Firefox's
            # mobile (Android) deployment target for the New Tab widget.
            context = browser.new_context(
                viewport={"width": 360, "height": 640},
                user_agent=("Mozilla/5.0 (Android 10; Mobile; rv:120.0) "
                            "Gecko/120.0 Firefox/120.0"),
                has_touch=True,
                is_mobile=False,  # Firefox/Playwright doesn't fully support is_mobile
                device_scale_factor=2.0,  # Moto G4 is ~xhdpi
                record_har_path=str(har_path),
                record_har_content="omit",
            )
            page = context.new_page()
            print(f"[{i:2d}/{len(TEST_PAGES)}] {url}", flush=True)
            t0 = time.time()
            info = crawl_one(page, url, har_path, type_log_path)
            elapsed = time.time() - t0
            page.close()
            context.close()  # this finalizes the HAR file

            # Count tracker requests in the HAR
            try:
                with open(har_path) as f:
                    har = json.load(f)
                info["n_requests"] = len(har.get("log", {}).get("entries", []))
            except Exception as e:
                info["har_parse_error"] = str(e)[:80]
            info["elapsed_s"] = round(elapsed, 1)
            print(f"   -> {'OK' if info['ok'] else 'FAIL'}: {info['n_requests']} requests, {info['elapsed_s']}s",
                  flush=True)
            summary.append(info)

        browser.close()

    summary_path = OUT_DIR / "_crawl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(1 for s in summary if s.get("ok"))
    total_req = sum(s.get("n_requests", 0) for s in summary)
    print(f"\n=== Crawl summary ===")
    print(f"Pages attempted: {len(summary)}")
    print(f"Pages succeeded: {n_ok}")
    print(f"Total requests captured: {total_req:,}")
    print(f"HARs written to: {OUT_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
