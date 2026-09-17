"""
Re-enable Remote Settings in the Playwright-bundled Firefox so that Enhanced
Tracking Protection actually blocks.

WHY THIS IS NEEDED
------------------
Firefox no longer ships the ETP tracker lists in the installer. They are
fetched at runtime through the pseudo-URL `moz-sbrs://antitracking`, which is
served by `UrlClassifierRemoteSettingsService`; that service reads
`RemoteSettings("tracking-protection-lists")`. So no Remote Settings, no
tracker lists, and ETP blocks nothing at all.

Playwright's Firefox build disables Remote Settings in three places. Two are
prefs in `playwright.cfg` and the crawler overrides them at launch
(`services.settings.server`, `browser.safebrowsing.provider.mozilla.updateURL`
-- see PRIVATE_PREFS in firefox_crawl_500_tracking.py). The third cannot be
overridden by any pref, because it is patched into the packaged JavaScript:

    get shouldSkipRemoteActivity() {
      // Playwright does not set Cu.isInAutomation, hence we just return true
      // here in order to disable the remote activity.
      return true;
      ...unreachable original body...
    }

With that `return true` in place, `RemoteSettingsClient.sync()` returns
immediately, the collection stays empty, and the list manager reports
"update success" having received zero records. The observable symptom is a
crawl where the ETP arm loads every tracker exactly like the control arm.

This script removes that single `return true`, restoring the upstream body
(which checks `Cu.isInAutomation` and enterprise policy, neither of which
applies here). Nothing else in the browser is modified.

SCOPE AND REVERSIBILITY
-----------------------
Only `omni.ja` inside the Playwright browser cache is touched -- never the
system Firefox, and nothing inside this repository. The original is saved
next to it as `omni.ja.pre-etp-patch` on first run, so `--revert` restores a
byte-identical browser. `playwright install --force firefox` also restores it,
and *will silently undo this patch*, so re-run `--check` after any Playwright
upgrade.

USAGE
-----
    python src/patch_playwright_firefox.py --check
    python src/patch_playwright_firefox.py --apply
    python src/patch_playwright_firefox.py --revert

`--apply` is idempotent. Applying the patch is necessary but not sufficient:
the crawler still runs a live canary page and refuses to record an ETP arm
that is not demonstrably blocking. See `--verify-etp` there.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# The exact text Playwright injects. We match the comment as well as the
# statement so that we cannot possibly strip an unrelated `return true`.
PLAYWRIGHT_BLOCK = """    // Playwright does not set Cu.isInAutomation, hence we just return true
    // here in order to disable the remote activity.
    return true;
"""

REPLACEMENT = (
    "    // [tracker-cost-model] Playwright's unconditional `return true` was\n"
    "    // removed here so that Remote Settings can sync the ETP tracker\n"
    "    // lists. See src/patch_playwright_firefox.py for the rationale.\n"
)

# Path of the file to patch inside omni.ja.
TARGET_MEMBER = "modules/services-settings/Utils.sys.mjs"

BACKUP_SUFFIX = ".pre-etp-patch"


def _registry_firefox_dir() -> Path | None:
    """Resolve the bundled Firefox directory from Playwright's registry.

    Reading `browsers.json` avoids spinning up the Node driver just to learn a
    path; starting and stopping it prints spurious asyncio teardown warnings.
    Returns None if the layout is not what we expect, and the caller falls
    back to asking Playwright directly.
    """
    import json
    import os

    try:
        import playwright
    except ImportError:
        return None

    manifest = Path(playwright.__file__).parent / "driver" / "package" / "browsers.json"
    try:
        entries = json.loads(manifest.read_text())["browsers"]
    except Exception:
        return None
    revision = next(
        (e["revision"] for e in entries if e.get("name") == "firefox"), None)
    if revision is None:
        return None

    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if root:
        base = Path(root)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "ms-playwright"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    else:
        base = Path.home() / ".cache" / "ms-playwright"

    d = base / f"firefox-{revision}"
    return d if d.exists() else None


def find_omni_ja() -> Path:
    """Locate omni.ja in the Playwright Firefox install.

    This must resolve to the same build the crawler launches, so it honours
    PLAYWRIGHT_BROWSERS_PATH and the pinned revision rather than globbing for
    whatever firefox-* directories happen to be lying around.
    """
    exe = None
    d = _registry_firefox_dir()
    if d is not None:
        for rel in ("firefox/Nightly.app/Contents/MacOS/firefox",  # macOS
                    "firefox/firefox",                             # Linux
                    "firefox/firefox.exe"):                        # Windows
            if (d / rel).exists():
                exe = d / rel
                break

    if exe is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            sys.exit("playwright is not installed in this interpreter.")
        with sync_playwright() as p:
            exe = Path(p.firefox.executable_path)

    if not exe.exists():
        sys.exit(
            f"Firefox executable not found at {exe}.\n"
            "Run: python -m playwright install firefox"
        )

    # macOS: .../firefox/Nightly.app/Contents/MacOS/firefox
    #        -> .../Contents/Resources/omni.ja
    # Linux: .../firefox/firefox -> .../firefox/omni.ja
    candidates = [
        exe.parent.parent / "Resources" / "omni.ja",  # macOS
        exe.parent / "omni.ja",                       # Linux / Windows
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"Could not find omni.ja near {exe}; looked in {candidates}")


def read_member(omni: Path, member: str) -> str:
    with zipfile.ZipFile(omni) as z:
        return z.read(member).decode("utf-8")


def patch_state(omni: Path) -> str:
    """Return 'playwright-disabled', 'patched', or 'unknown'."""
    src = read_member(omni, TARGET_MEMBER)
    if PLAYWRIGHT_BLOCK in src:
        return "playwright-disabled"
    if "[tracker-cost-model]" in src:
        return "patched"
    return "unknown"


def rewrite_omni(omni: Path, new_source: str) -> None:
    """Rewrite omni.ja with one member replaced.

    zipfile cannot edit in place, so we stream every entry into a new archive
    and swap it in. Mozilla's build step aligns omni.ja for mmap; that is a
    startup optimisation only, and Firefox reads an ordinary deflated zip
    correctly, which is what we produce here.
    """
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "omni.ja"
        with zipfile.ZipFile(omni) as src, zipfile.ZipFile(
            staged, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                if item.filename == TARGET_MEMBER:
                    dst.writestr(item, new_source)
                else:
                    dst.writestr(item, src.read(item.filename))
        shutil.move(str(staged), str(omni))


def cmd_check(omni: Path) -> int:
    state = patch_state(omni)
    backup = omni.with_name(omni.name + BACKUP_SUFFIX)
    print(f"omni.ja: {omni}")
    print(f"backup:  {backup if backup.exists() else '(none)'}")
    if state == "patched":
        print("state:   PATCHED -- Remote Settings sync is enabled.")
        return 0
    if state == "playwright-disabled":
        print("state:   UNPATCHED -- Playwright disables Remote Settings;")
        print("         ETP will block nothing. Run with --apply.")
        return 1
    print("state:   UNKNOWN -- neither Playwright's block nor our patch marker")
    print("         was found. The build may have changed; inspect")
    print(f"         {TARGET_MEMBER} before proceeding.")
    return 2


def cmd_apply(omni: Path) -> int:
    state = patch_state(omni)
    if state == "patched":
        print("Already patched; nothing to do.")
        return 0
    if state == "unknown":
        print("Refusing to patch: could not find Playwright's "
              "`shouldSkipRemoteActivity` override.")
        print("The bundled Firefox has probably changed. Re-read "
              f"{TARGET_MEMBER} and update PLAYWRIGHT_BLOCK.")
        return 2

    backup = omni.with_name(omni.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(omni, backup)
        print(f"Backed up original -> {backup}")

    src = read_member(omni, TARGET_MEMBER)
    patched = src.replace(PLAYWRIGHT_BLOCK, REPLACEMENT, 1)
    if patched == src:
        return 2  # unreachable given the state check, but do not write blindly
    rewrite_omni(omni, patched)

    if patch_state(omni) != "patched":
        print("Patch did not verify after writing; restoring backup.")
        shutil.copy2(backup, omni)
        return 2
    print("Patched. Remote Settings sync is now enabled in "
          "Playwright's Firefox.")
    print("Next: verify ETP really blocks with")
    print("  python src/firefox_crawl_500_tracking.py --verify-etp")
    return 0


def cmd_revert(omni: Path) -> int:
    backup = omni.with_name(omni.name + BACKUP_SUFFIX)
    if not backup.exists():
        print(f"No backup at {backup}; nothing to revert.")
        print("To restore a pristine browser: "
              "python -m playwright install --force firefox")
        return 1
    shutil.copy2(backup, omni)
    print(f"Restored original omni.ja from {backup}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="Report whether the patch is applied.")
    g.add_argument("--apply", action="store_true",
                   help="Apply the patch (idempotent).")
    g.add_argument("--revert", action="store_true",
                   help="Restore the original omni.ja from backup.")
    args = ap.parse_args()

    omni = find_omni_ja()
    if args.check:
        return cmd_check(omni)
    if args.apply:
        return cmd_apply(omni)
    return cmd_revert(omni)


if __name__ == "__main__":
    sys.exit(main())
