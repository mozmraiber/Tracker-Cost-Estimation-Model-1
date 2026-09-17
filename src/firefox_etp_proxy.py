"""
Offline reimplementation of "would Firefox's ETP block this request?", built
from the same data the browser uses.

A plain `is_tracker(url)` lookup overstates ETP badly -- on the paired crawl it
flags 4.4x more requests than Firefox actually blocked -- because being on the
tracker list is only the first of four tests Firefox applies. This module adds
the other three, so an offline savings estimate can be scoped to the requests
that would really be blocked.

The four tests, in the order they are applied:

  1. the request host is on the tracking list, in a category the active
     protection level covers                       disconnect-blacklist.json
  2. the request is third-party to the top-level page
  3. the tracker is not owned by the same entity as the page
                                                  disconnect-entitylist.json
  4. no allowlist exception covers this (request, page) pair
                     url-classifier-exceptions.json, url-classifier-skip-urls.json

Data provenance, all under data/external/firefox_lists/:
  disconnect-blacklist.json       mozilla-services/shavar-prod-lists -- the
                                  source the *-track-digest256 lists are built
                                  from, so it is what Firefox blocks against
                                  rather than a differently-scoped snapshot
  disconnect-entitylist.json      same repo; the source of
                                  mozstd-trackwhite-digest256
  url-classifier-exceptions.json  Remote Settings, main/url-classifier-exceptions
  url-classifier-skip-urls.json   Firefox's in-tree dump of the same collection
                                  family, for the annotation skip list

Refresh with src/fetch_firefox_lists.py.

WHAT THIS DOES NOT DO
---------------------
The shipped digest256 lists are SHA-256 hashes, so they cannot be enumerated;
this works from the upstream source lists instead. Those are rebuilt into
digests on Mozilla's schedule, so a freshly published tracker can be on the
source list before it is in a browser's list, or vice versa.

`filter_expression` version gating is evaluated only for the
`versionCompare(...)` comparisons the collection actually uses, against
`--firefox-version`. Anything else is treated as applicable.

Private browsing matters: 599 of the 1,201 exceptions are
`isPrivateBrowsingOnly`. A real Private Window honours them; the crawl's
blocking arm is a normal window with `privacy.trackingprotection.enabled`
forced on, so it does *not*, and would block slightly more than real PBM.
`pbm=False` (the default) reproduces the crawl; pass `pbm=True` to model a real
Private Window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# Categories ETP Standard's tracking-protection feature blocks. Content is
# Strict-only; the fingerprinting and cryptomining categories are handled by
# separate features that the paired crawl left on in *both* arms.
BASE_CATEGORIES = ("Advertising", "Analytics", "Social")
STRICT_CATEGORIES = BASE_CATEGORIES + ("Content",)

DEFAULT_LIST_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "firefox_lists"


def host_of(url: str) -> str:
    if "//" not in url:
        url = "http://" + url
    return urlsplit(url).netloc.lower().split(":")[0].rstrip(".")


def host_matches(host: str, listed: str) -> bool:
    """Firefox matches a listed domain and everything under it."""
    listed = listed.lower().lstrip(".").rstrip(".")
    return host == listed or host.endswith("." + listed)


def _glob_to_re(pattern: str) -> re.Pattern:
    """Compile a `*://*.example.com/*` match pattern.

    Only `*` is a wildcard in this collection, so escaping everything else and
    widening `*` is faithful; the records ship apex and subdomain variants
    separately, so no extra subdomain logic is needed.
    """
    return re.compile("^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$",
                      re.IGNORECASE)


def _version_key(v: str) -> tuple:
    """Sortable key for a Firefox version like `142.0a1`.

    A pre-release suffix sorts below the same release, which is all the
    precision these comparisons need.
    """
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:([ab])(\d+))?", v.strip())
    if not m:
        return (0,)
    major, minor, pre, pren = m.groups()
    return (int(major), int(minor or 0), 0 if pre else 1,
            {"a": 0, "b": 1}.get(pre or "", 2), int(pren or 0))


_VC_RE = re.compile(
    r'env\.version\s*\|\s*versionCompare\(\s*"([^"]+)"\s*\)\s*(<=|>=|<|>|==|!=)\s*0')


def filter_expression_applies(expr: str | None, version: str) -> bool:
    """Evaluate the `versionCompare` clauses of a filter expression.

    Every clause the collection currently uses is a `versionCompare(...)`
    against 0, combined with `&&`. Clauses of any other shape are ignored
    rather than guessed at, which errs toward treating a record as applicable.
    """
    if not expr:
        return True
    ours = _version_key(version)
    for target, op in _VC_RE.findall(expr):
        theirs = _version_key(target)
        cmp = (ours > theirs) - (ours < theirs)
        ok = {"<": cmp < 0, "<=": cmp <= 0, ">": cmp > 0, ">=": cmp >= 0,
              "==": cmp == 0, "!=": cmp != 0}[op]
        if not ok:
            return False
    return True


class FirefoxETPProxy:
    """Decides whether Firefox's ETP would block a request."""

    def __init__(self, list_dir: Path | str = DEFAULT_LIST_DIR, *,
                 strict: bool = False, pbm: bool = False,
                 baseline_allowlist: bool = True,
                 convenience_allowlist: bool = True,
                 firefox_version: str = "153.0"):
        self.dir = Path(list_dir)
        self.categories = STRICT_CATEGORIES if strict else BASE_CATEGORIES
        self.pbm = pbm
        self.baseline_allowlist = baseline_allowlist
        self.convenience_allowlist = convenience_allowlist
        self.version = firefox_version

        self._load_blacklist()
        self._load_entitylist()
        self._load_exceptions()
        self._load_skip_urls()

    # -- data loading ----------------------------------------------------
    def _load_blacklist(self) -> None:
        raw = json.loads((self.dir / "disconnect-blacklist.json").read_text())
        # categories -> [ {Org: {url: [domains...]}} ]
        self.tracker_domains: set[str] = set()
        self.domain_category: dict[str, str] = {}
        for cat in self.categories:
            for org_entry in raw["categories"].get(cat, []):
                for _org, sites in org_entry.items():
                    if not isinstance(sites, dict):
                        continue
                    for _site, domains in sites.items():
                        if not isinstance(domains, list):
                            continue
                        for d in domains:
                            d = str(d).lower().lstrip(".")
                            self.tracker_domains.add(d)
                            self.domain_category.setdefault(d, cat)

    def _load_entitylist(self) -> None:
        raw = json.loads((self.dir / "disconnect-entitylist.json").read_text())
        ents = raw.get("entities", raw)
        # resource domain -> list of entities claiming it, each with properties
        self.resource_entities: dict[str, list[tuple[str, ...]]] = {}
        for name, spec in ents.items():
            props = tuple(str(p).lower() for p in spec.get("properties", []))
            for res in spec.get("resources", []):
                self.resource_entities.setdefault(
                    str(res).lower(), []).append(props)
        self._n_entities = len(ents)

    def _load_exceptions(self) -> None:
        raw = json.loads((self.dir / "url-classifier-exceptions.json").read_text())
        self.exceptions: list[dict] = []
        for r in raw.get("data", []):
            if "tracking-protection" not in (r.get("classifierFeatures") or []):
                continue
            if not filter_expression_applies(r.get("filter_expression"), self.version):
                continue
            cat = r.get("category")
            if cat == "baseline" and not self.baseline_allowlist:
                continue
            if cat == "convenience" and not self.convenience_allowlist:
                continue
            # A PBM-only exception does nothing outside a Private Window.
            if r.get("isPrivateBrowsingOnly") and not self.pbm:
                continue
            pat = r.get("urlPattern")
            if not pat:
                continue
            self.exceptions.append({
                "url_re": _glob_to_re(pat),
                "top_re": (_glob_to_re(r["topLevelUrlPattern"])
                           if r.get("topLevelUrlPattern") else None),
                "category": cat,
            })

    def _load_skip_urls(self) -> None:
        self.skip_hosts: list[str] = []
        f = self.dir / "url-classifier-skip-urls.json"
        if not f.exists():
            return
        for r in json.loads(f.read_text()).get("data", []):
            if r.get("feature") != "tracking-protection":
                continue
            if not filter_expression_applies(r.get("filter_expression"), self.version):
                continue
            if r.get("pattern"):
                self.skip_hosts.append(str(r["pattern"]).lower())

    # -- the decision ----------------------------------------------------
    def is_listed(self, url: str) -> bool:
        host = host_of(url)
        if host in self.tracker_domains:
            return True
        # Walk parent domains: a listed domain covers its subdomains.
        parts = host.split(".")
        return any(".".join(parts[i:]) in self.tracker_domains
                   for i in range(1, len(parts) - 1))

    def is_third_party(self, url: str, page_url: str) -> bool:
        h, p = host_of(url), host_of(page_url)
        if not h or not p:
            return False
        return not (h == p or h.endswith("." + p) or p.endswith("." + h))

    def same_entity(self, url: str, page_url: str) -> bool:
        """Does one entity own both the tracker resource and the page?"""
        h, p = host_of(url), host_of(page_url)
        parts = h.split(".")
        candidates = [h] + [".".join(parts[i:]) for i in range(1, len(parts) - 1)]
        for cand in candidates:
            for props in self.resource_entities.get(cand, ()):
                if any(host_matches(p, prop) for prop in props):
                    return True
        return False

    def exception_applies(self, url: str, page_url: str) -> str | None:
        host = host_of(url)
        for skip in self.skip_hosts:
            if host_matches(host, skip):
                return "skip-url"
        for e in self.exceptions:
            if not e["url_re"].match(url):
                continue
            if e["top_re"] is not None and not e["top_re"].match(page_url):
                continue
            return f"exception:{e['category']}"
        return None

    def decide(self, url: str, page_url: str) -> tuple[bool, str]:
        """(would_block, reason). The reason names the test that decided it."""
        if not self.is_listed(url):
            return False, "not-listed"
        if not self.is_third_party(url, page_url):
            return False, "first-party"
        if self.same_entity(url, page_url):
            return False, "same-entity"
        if (why := self.exception_applies(url, page_url)) is not None:
            return False, why
        return True, "blocked"

    def would_block(self, url: str, page_url: str) -> bool:
        return self.decide(url, page_url)[0]

    def describe(self) -> dict:
        return {
            "categories": list(self.categories),
            "tracker_domains": len(self.tracker_domains),
            "entities": self._n_entities,
            "entity_resources": len(self.resource_entities),
            "active_exceptions": len(self.exceptions),
            "skip_hosts": len(self.skip_hosts),
            "pbm": self.pbm,
            "firefox_version": self.version,
        }


if __name__ == "__main__":
    proxy = FirefoxETPProxy()
    print(json.dumps(proxy.describe(), indent=2))
    checks = [
        ("https://static.criteo.net/js/ld/ld.js", "https://www.dailymail.co.uk/"),
        ("https://i.ebayimg.com/x.jpg", "https://ebay.com/"),
        ("https://img.ltwebstatic.com/a.png", "https://shein.com/"),
        ("https://a0.muscache.com/x.jpg", "https://airbnb.com/"),
        ("https://www.google-analytics.com/collect", "https://www.bbc.com/"),
        ("https://media.net/lib.js", "https://media.net/"),
    ]
    for url, page in checks:
        block, why = proxy.decide(url, page)
        print(f"  {'BLOCK' if block else 'allow':>5}  {why:<20} {url[:52]}  on {page}")
