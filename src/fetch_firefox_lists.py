"""
Download the lists Firefox's ETP actually consults, for src/firefox_etp_proxy.py.

Four files, from two kinds of source:

  disconnect-blacklist.json       mozilla-services/shavar-prod-lists. The
  disconnect-entitylist.json      source the *-track-digest256 and
                                  mozstd-trackwhite-digest256 lists are built
                                  from. The shipped lists themselves are
                                  SHA-256 digests and cannot be enumerated,
                                  so the upstream source is the only readable
                                  form.

  url-classifier-exceptions.json  Remote Settings, which serves the live
                                  allowlist the browser reads. Public, no auth.

  url-classifier-skip-urls.json   copied from a local mozilla-central checkout
                                  if one is given, since Firefox ships this
                                  collection as an in-tree dump.

Provenance matters for reproducing a measurement, so a sidecar
`_provenance.json` records the URLs, fetch time, sizes and sha256 of whatever
was written.

Usage:
    python src/fetch_firefox_lists.py
    python src/fetch_firefox_lists.py --mozilla-central ~/firefox
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SHAVAR_BASE = ("https://raw.githubusercontent.com/mozilla-services/"
               "shavar-prod-lists/master")
REMOTE_SETTINGS = ("https://firefox.settings.services.mozilla.com/v1/buckets/"
                   "main/collections/{collection}/records")

DOWNLOADS = {
    "disconnect-blacklist.json": f"{SHAVAR_BASE}/disconnect-blacklist.json",
    "disconnect-entitylist.json": f"{SHAVAR_BASE}/disconnect-entitylist.json",
    "url-classifier-exceptions.json":
        REMOTE_SETTINGS.format(collection="url-classifier-exceptions"),
}

# Shipped as an in-tree Remote Settings dump rather than fetched at runtime.
IN_TREE = {
    "url-classifier-skip-urls.json":
        "services/settings/dumps/main/url-classifier-skip-urls.json",
}

DEFAULT_OUT = (Path(__file__).resolve().parents[1]
               / "data" / "external" / "firefox_lists")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mozilla-central", default=None,
                    help="Path to a mozilla-central checkout, for the in-tree "
                         "dump. Skipped if not given.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict] = {}

    for name, url in DOWNLOADS.items():
        dest = out / name
        print(f"fetching {name} ...", flush=True)
        req = urllib.request.Request(
            url, headers={"User-Agent": "tracker-cost-model/fetch-lists"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
        # Fail loudly rather than leave a half-written list in place: a
        # truncated allowlist silently changes every downstream number.
        json.loads(body)
        dest.write_bytes(body)
        provenance[name] = {"source": url, "bytes": len(body),
                            "sha256_16": _sha256(dest)}
        print(f"  {len(body):,} bytes -> {dest}")

    if args.mozilla_central:
        root = Path(args.mozilla_central).expanduser()
        for name, rel in IN_TREE.items():
            src = root / rel
            if not src.exists():
                print(f"  skipped {name}: not found at {src}")
                continue
            shutil.copy2(src, out / name)
            provenance[name] = {"source": str(src),
                                "bytes": (out / name).stat().st_size,
                                "sha256_16": _sha256(out / name)}
            print(f"  copied {name} from {src}")
    else:
        print("  (no --mozilla-central given; url-classifier-skip-urls.json "
              "not refreshed)")

    (out / "_provenance.json").write_text(json.dumps({
        "fetched_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": provenance,
    }, indent=2))
    print(f"\nWrote provenance to {out / '_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
