"""An independent implementation of the Disconnect matching rules.

Written straight against the services JSON, with no shared code with the C++
matcher, so that both the DuckDB extension (test_disconnect.py) and the Python
module (test_module.py) can be cross-checked against it.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_JSON = os.path.join(HERE, "..", "..", "data", "external", "disconnect_services.json")

# Must match EXCLUDED_BY_DEFAULT in src/include/disconnect_list.hpp: the
# category an unfiltered is_tracker() leaves out.
EXCLUDED_BY_DEFAULT = "Content"

# Must match CATEGORY_PRIORITY in src/disconnect_list.cpp.
CATEGORY_PRIORITY = [
    "Advertising",
    "Analytics",
    "Social",
    "Cryptomining",
    "FingerprintingInvasive",
    "Content",
    "EmailAggressive",
    "Email",
    "Anti-fraud",
    "ConsentManagers",
    "FingerprintingGeneral",
]

class ReferenceMatcher:
    """A straightforward Python implementation of the same matching rules."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.categories = list(document["categories"].keys())
        self.hosts = {}       # host -> list of (category, organization)
        self.paths = {}       # host -> list of (path, category, organization)
        for category, entries in document["categories"].items():
            for entry in entries:
                for org, sites in entry.items():
                    for _, patterns in sites.items():
                        if not isinstance(patterns, list):
                            continue
                        for pattern in patterns:
                            pattern = pattern.strip().lower()
                            host, _, path = pattern.partition("/")
                            host = host.rstrip(".")
                            if not host:
                                continue
                            if path:
                                self.paths.setdefault(host, []).append(("/" + path, category, org))
                            else:
                                self.hosts.setdefault(host, []).append((category, org))

    def default_categories(self):
        """What an unfiltered is_tracker() matches under."""
        return {c for c in self.categories if c != EXCLUDED_BY_DEFAULT}

    def is_tracker(self, url, allowed=None):
        """The verdict is_tracker() gives: no category filter means every
        category but EXCLUDED_BY_DEFAULT, which is not the same as `allowed`
        being None in `match` — that stays every category, because it is what
        the describing functions look up under."""
        if allowed is None:
            allowed = self.default_categories()
        return self.match(url, allowed=allowed) is not None

    @staticmethod
    def split(url):
        rest = url.strip()
        scheme = rest.find("://")
        if scheme != -1 and all(c not in rest[:scheme] for c in "/?#"):
            rest = rest[scheme + 3:]
        elif rest.startswith("//"):
            rest = rest[2:]
        cut = len(rest)
        for sep in "/?#":
            position = rest.find(sep)
            if position != -1:
                cut = min(cut, position)
        authority, remainder = rest[:cut], rest[cut:]
        authority = authority.rpartition("@")[2]
        if authority.startswith("["):
            authority = authority[1:authority.find("]")] if "]" in authority else authority[1:]
        else:
            host, sep, port = authority.rpartition(":")
            if sep and port.isdigit():
                authority = host
        path = ""
        if remainder.startswith("/"):
            path = remainder.split("?")[0].split("#")[0]
        return authority.lower().rstrip("."), path.lower()

    def match(self, url, allowed=None):
        """Return (category, organization, pattern) or None."""
        host, path = self.split(url)
        while host:
            for rule_path, category, org in self.paths.get(host, []):
                if path.startswith(rule_path) and (allowed is None or category in allowed):
                    return category, org, host + rule_path
            listed = [(c, o) for c, o in self.hosts.get(host, []) if allowed is None or c in allowed]
            if listed:
                listed.sort(key=lambda item: CATEGORY_PRIORITY.index(item[0]))
                return listed[0][0], listed[0][1], host
            _, dot, host = host.partition(".")
            if not dot:
                break
        return None

    def categories_of(self, url):
        host, path = self.split(url)
        result = self.match(url)
        if result is None:
            return None
        matched = result[2]
        if "/" in matched:
            rule_path = "/" + matched.partition("/")[2]
            host_key = matched.partition("/")[0]
            found = {c for p, c, _ in self.paths.get(host_key, []) if p == rule_path}
        else:
            found = {c for c, _ in self.hosts.get(matched, [])}
        return [c for c in self.categories if c in found]
