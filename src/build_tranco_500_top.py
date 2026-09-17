"""
Build the top-N reachable publisher URL list from a Tranco daily list.

Inputs:  Tranco CSV (rank,domain — no header).
Outputs: text file with one https URL per line, plus a `.summary.txt`
         listing failures with reasons.

Filtering:
  - HTTPS HEAD request, 10s timeout, follows up to 3 redirects
  - Status 200 → keep; any other outcome → drop with reason
  - Falls back to GET on 405/501 (some servers reject HEAD)
  - 200ms between probes (polite rate limiting)
  - Stops as soon as the target reachable count is hit

The Tranco list ID and download date must be captured in §6.3 of
paper.tex *before* this script runs (pre-registration discipline).
This script does not enforce that; it just builds the list.
"""

from __future__ import annotations
import argparse
import csv
import time
import urllib.error
import urllib.request
from pathlib import Path


def _opener_with_redirect_cap(max_redirects: int):
    class CappedRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_302(self, req, fp, code, msg, headers):
            req._n = getattr(req, "_n", 0) + 1
            if req._n > max_redirects:
                raise urllib.error.HTTPError(
                    req.full_url, code, "too many redirects", headers, fp
                )
            return super().http_error_302(req, fp, code, msg, headers)
        http_error_301 = http_error_302
        http_error_303 = http_error_302
        http_error_307 = http_error_302
        http_error_308 = http_error_302

    return urllib.request.build_opener(CappedRedirect)


def head_check(url: str, timeout: float = 10.0,
               max_redirects: int = 3) -> tuple[bool, str]:
    """Return (reachable, reason). Tries HEAD; falls back to GET on 405/501."""
    opener = _opener_with_redirect_cap(max_redirects)
    headers = {"User-Agent": "Mozilla/5.0 (Tranco-reachability-check)"}

    def _try(method: str) -> tuple[bool, str]:
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with opener.open(req, timeout=timeout) as resp:
                return resp.status == 200, str(resp.status)
        except urllib.error.HTTPError as e:
            return False, f"http_{e.code}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:60]}"

    ok, reason = _try("HEAD")
    if not ok and reason in ("http_405", "http_501"):
        ok, reason = _try("GET")
    return ok, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranco", required=True,
                    help="Path to Tranco CSV (rank,domain).")
    ap.add_argument("--out", required=True,
                    help="Output text file: one URL per line.")
    ap.add_argument("--target", type=int, default=500,
                    help="Number of reachable URLs to collect.")
    ap.add_argument("--rate-ms", type=int, default=200,
                    help="Milliseconds between probes.")
    args = ap.parse_args()

    tranco_path = Path(args.tranco)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    domains = []
    with open(tranco_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            domains.append(row[1] if len(row) >= 2 else row[0])

    print(f"Loaded {len(domains):,} domains from {tranco_path}")
    print(f"Probing for {args.target} reachable HTTPS endpoints "
          f"({args.rate_ms} ms between probes)...")

    rate_s = args.rate_ms / 1000.0
    reachable: list[str] = []
    failures: list[tuple[str, str]] = []
    n_checked = 0

    for dom in domains:
        if len(reachable) >= args.target:
            break
        n_checked += 1
        url = f"https://{dom}/"
        ok, reason = head_check(url)
        if ok:
            reachable.append(url)
            print(f"[{n_checked:4d}] OK   "
                  f"({len(reachable):3d}/{args.target})  {dom}", flush=True)
        else:
            failures.append((dom, reason))
            print(f"[{n_checked:4d}] FAIL ({reason})  {dom}", flush=True)
        time.sleep(rate_s)

    with open(out_path, "w") as f:
        for u in reachable:
            f.write(u + "\n")

    summary_path = out_path.with_suffix(out_path.suffix + ".summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Tranco source:       {tranco_path}\n")
        f.write(f"Target reachable:    {args.target}\n")
        f.write(f"Domains checked:     {n_checked}\n")
        f.write(f"Reachable URLs:      {len(reachable)}\n")
        f.write(f"Failed/skipped:      {len(failures)}\n\n")
        f.write("--- Failures (domain, reason) ---\n")
        for dom, why in failures:
            f.write(f"{dom}\t{why}\n")

    print()
    print(f"Wrote {len(reachable)} URLs to {out_path}")
    print(f"Wrote summary to        {summary_path}")
    if len(reachable) < args.target:
        print(f"WARNING: target was {args.target} but only "
              f"{len(reachable)} reachable URLs found.")


if __name__ == "__main__":
    main()
