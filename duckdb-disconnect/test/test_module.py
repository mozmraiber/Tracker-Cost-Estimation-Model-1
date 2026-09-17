#!/usr/bin/env python3
"""Test suite for the `disconnect` Python module.

Run with: make test-module

The module and the DuckDB extension share their matcher, so this suite focuses
on the Python calling layer -- NULL handling, category filters, the batch
entry point, error types -- and then cross-checks every answer against the same
independent reference matcher the extension is checked against.
"""

import json
import os
import pathlib
import sys
import tempfile
import threading

import disconnect

from reference_matcher import LIST_JSON, ReferenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def check_equal(actual, expected, message):
    check(actual == expected, "%s (got %r, expected %r)" % (message, actual, expected))


def check_raises(exception, body, message):
    try:
        body()
    except exception:
        check(True, message)
        return
    except BaseException as e:
        check(False, "%s (raised %s instead)" % (message, type(e).__name__))
        return
    check(False, "%s (nothing raised)" % message)


SCALARS = [
    disconnect.is_tracker,
    disconnect.tracker_category,
    disconnect.tracker_categories,
    disconnect.tracker_owner,
    disconnect.tracker_owner_url,
    disconnect.tracker_pattern,
    disconnect.tracker_index,
    disconnect.url_host,
    disconnect.match,
]


def test_basics():
    check(disconnect.is_tracker("https://www.google-analytics.com/collect") is True,
          "google-analytics is a tracker")
    check(disconnect.is_tracker("https://example.com/") is False, "example.com is not a tracker")
    check(disconnect.is_tracker("") is False, "empty string is not a tracker")
    check_equal(disconnect.tracker_category("https://connect.facebook.net/en_US/fbevents.js"), "Social",
                "facebook is Social")
    check_equal(disconnect.tracker_owner("connect.facebook.net"), "Meta", "facebook is owned by Meta")
    check_equal(disconnect.tracker_pattern("https://stats.g.doubleclick.net/x"), "doubleclick.net",
                "parent domains match")
    check_equal(disconnect.entries()[disconnect.tracker_index("https://stats.g.doubleclick.net/x")]["pattern"],
                "doubleclick.net", "tracker_index points at the entry that matched")
    check(disconnect.tracker_category("https://example.com/") is None, "no category without a match")
    check(disconnect.tracker_index("https://example.com/") is None, "no index without a match")
    check(disconnect.tracker_categories("https://example.com/") is None, "no category list either")
    check(disconnect.match("https://example.com/") is None, "no match dict either")
    check_equal(disconnect.url_host("HTTPS://user:pw@Example.COM.:8443/a?b"), "Example.COM",
                "url_host strips userinfo, port and root dot but not case")
    check(disconnect.url_host("https:///path") is None, "no host is None")
    check_equal(disconnect.__version__, "0.1.0", "module reports its version")

    # Odd but not invalid input must not crash or match.
    for url in ["not a url", "http://", "https://:8080/", "///", "@", ":", "..", "http://.../x", "%%%",
                "https://" + "a" * 5000 + ".com/", "☃.example", "a\x00b"]:
        for function in SCALARS:
            function(url)
    check(True, "odd input is handled")


def test_null_propagates():
    for function in SCALARS:
        check(function(None) is None, "%s(None) is None" % function.__name__)
    check_equal(disconnect.is_tracker_many([None, "doubleclick.net", None]), [None, True, None],
                "None propagates through a batch")


def test_keyword_arguments():
    check(disconnect.is_tracker(url="doubleclick.net") is True, "url is a keyword")
    check(disconnect.is_tracker(url="google-analytics.com", categories="Analytics") is True,
          "categories is a keyword")
    check_equal(disconnect.is_tracker_many(urls=["doubleclick.net"], categories=["Advertising"]), [True],
                "is_tracker_many takes keywords")
    check_equal(disconnect.tracker_owner(url="doubleclick.net"), "Google", "tracker_owner takes a keyword")


def test_category_filters():
    url = "https://www.google-analytics.com/x"
    check(disconnect.is_tracker(url, "Analytics") is True, "filter selects")
    check(disconnect.is_tracker(url, "Advertising") is False, "filter excludes")
    check(disconnect.is_tracker(url, "analytics") is True, "filter is case-insensitive")
    check(disconnect.is_tracker(url, " Advertising , Analytics ") is True, "filter tolerates whitespace")
    check(disconnect.is_tracker(url, ["Advertising", "Analytics"]) is True, "filter may be a sequence")
    check(disconnect.is_tracker(url, ("Advertising",)) is False, "a sequence filter also excludes")
    # The Content category lists first-party properties of tracking companies,
    # which a browser does not block, so an unfiltered call leaves it out.
    google = "https://www.google.com/search?q=x"
    check(disconnect.is_tracker(google) is False, "unfiltered does not match Content")
    check(disconnect.is_tracker(google, "Content") is True, "naming Content explicitly matches")
    check(disconnect.is_tracker_many([google]) == [False], "the batch form agrees")
    check(disconnect.is_tracker_many([google], "Content") == [True], "so does its filtered form")
    check(disconnect.is_tracker(google, "Advertising,Analytics,Social,Cryptomining") is False,
          "an ETP-standard-ish filter does not match google.com either")
    # Only the verdict changes; the classification is still there to be read.
    check(disconnect.tracker_category(google) == "Content", "tracker_category still reports Content")
    check(disconnect.tracker_owner(google) == "Google", "tracker_owner still reports the owner")
    check(disconnect.match(google)["category"] == "Content", "match() still describes it")

    for bad in ["NotACategory", "", "  ", "Analytics,Nope", [], ["Nope"]]:
        check_raises(ValueError, lambda bad=bad: disconnect.is_tracker(url, bad),
                     "filter %r is rejected" % (bad,))
        check_raises(ValueError, lambda bad=bad: disconnect.is_tracker_many([url], bad),
                     "filter %r is rejected in a batch" % (bad,))
    # A bad filter is an error even when there is nothing to classify.
    check_raises(ValueError, lambda: disconnect.is_tracker(None, "Nope"),
                 "a bad filter is rejected before NULL short-circuits")
    check_raises(ValueError, lambda: disconnect.is_tracker_many([], "Nope"),
                 "a bad filter is rejected on an empty batch")


def test_type_errors():
    for bad in [42, 3.5, b"doubleclick.net", ["doubleclick.net"], object()]:
        for function in SCALARS:
            check_raises(TypeError, lambda f=function, bad=bad: f(bad),
                         "%s rejects %s" % (function.__name__, type(bad).__name__))
    check_raises(TypeError, lambda: disconnect.is_tracker_many(["ok", 42]), "is_tracker_many rejects a non-str item")
    check_raises(TypeError, lambda: disconnect.is_tracker_many(42), "is_tracker_many rejects a non-iterable")
    check_raises(TypeError, lambda: disconnect.is_tracker("x", 42), "a non-sequence filter is rejected")
    check_raises(TypeError, lambda: disconnect.is_tracker("x", [42]), "a non-str category name is rejected")
    check_raises(TypeError, lambda: disconnect.is_tracker(), "is_tracker needs a url")


def probe_urls(reference):
    """The same probe set the extension is cross-checked over."""
    urls = []
    for host in list(reference.hosts) + list(reference.paths):
        urls.append("https://%s/some/path?q=1" % host)
        urls.append("https://sub.%s/x" % host)
        urls.append("https://not%s/x" % host)
        urls.append("HTTPS://%s.:443/X" % host.upper())
    for host, rules in reference.paths.items():
        for path, _, _ in rules:
            urls.append("https://www.%s%s/extra?a=b" % (host, path))
            urls.append("https://www.%s/other%s" % (host, path))
    return urls


def test_against_reference(reference, urls):
    mismatches = []
    for url in urls:
        actual = (disconnect.is_tracker(url), disconnect.tracker_category(url), disconnect.tracker_owner(url),
                  disconnect.tracker_pattern(url), disconnect.tracker_categories(url))
        expected = reference.match(url)
        if expected is None:
            wanted = (False, None, None, None, None)
        else:
            category, org, pattern = expected
            # is_tracker() answers under the default categories, the describing
            # functions under every category, so a Content-only host is
            # described in full and still answers False.
            wanted = (reference.is_tracker(url), category, org, pattern, reference.categories_of(url))
        if actual != wanted:
            mismatches.append((url, wanted, actual))
    check(not mismatches, "reference mismatches: %s" % mismatches[:5])
    print("  cross-checked %d URLs against the reference matcher" % len(urls))


def test_match_agrees_with_the_scalars(urls):
    """match() must report exactly what the one-field functions report.

    Keyed on tracker_category() rather than is_tracker(), because match() and
    the one-field functions all describe what the list says, including the
    Content category that is_tracker() declines to call a tracker.
    """
    mismatches = []
    for url in urls:
        result = disconnect.match(url)
        if disconnect.tracker_category(url) is None:
            if result is not None:
                mismatches.append((url, result))
            continue
        wanted = {
            "pattern": disconnect.tracker_pattern(url),
            "category": disconnect.tracker_category(url),
            "categories": disconnect.tracker_categories(url),
            "owner": disconnect.tracker_owner(url),
            "owner_url": disconnect.tracker_owner_url(url),
        }
        if result != wanted:
            mismatches.append((url, wanted, result))
    check(not mismatches, "match() disagrees with the scalars: %s" % mismatches[:5])


def test_index_points_into_entries(urls):
    """The entry tracker_index() names must be the one the scalars describe."""
    entries = disconnect.entries()
    mismatches = []
    for url in urls:
        index = disconnect.tracker_index(url)
        if index is None:
            if disconnect.is_tracker(url):
                mismatches.append((url, None))
            continue
        entry = entries[index]
        wanted = {
            "pattern": disconnect.tracker_pattern(url),
            "category": disconnect.tracker_category(url),
            "organization": disconnect.tracker_owner(url),
            "organization_url": disconnect.tracker_owner_url(url),
        }
        if entry != wanted:
            mismatches.append((url, index, wanted, entry))
    check(not mismatches, "tracker_index() disagrees with the scalars: %s" % mismatches[:5])
    # Every entry has to be reachable through its own pattern, and the index of
    # a multi-category host must be the entry carrying its primary category.
    unreachable = [e["pattern"] for e in entries if disconnect.tracker_index("https://" + e["pattern"]) is None]
    check(not unreachable, "patterns with no index: %s" % unreachable[:5])
    facebook = entries[disconnect.tracker_index("connect.facebook.net")]
    check_equal(facebook["category"], "Social", "the index prefers the primary category")


def test_filtered_against_reference(reference, urls):
    allowed = ["Advertising", "Analytics", "Social", "Cryptomining"]
    expected = [reference.match(url, allowed=set(allowed)) is not None for url in urls]
    check_equal(disconnect.is_tracker_many(urls, allowed), expected, "filtered batch agrees with the reference")
    mismatches = [url for url, want in zip(urls, expected) if disconnect.is_tracker(url, allowed) != want]
    check(not mismatches, "filtered scalar mismatches: %s" % mismatches[:5])


def test_batch_matches_the_scalar(urls):
    expected = [disconnect.is_tracker(url) for url in urls]
    check_equal(disconnect.is_tracker_many(urls), expected, "a batch agrees with the scalar")
    # More rows than the internal chunk size, and every input type of iterable.
    long_input = urls * 3
    check_equal(disconnect.is_tracker_many(long_input), expected * 3, "batching spans several chunks")
    check_equal(disconnect.is_tracker_many(tuple(urls[:10])), expected[:10], "a tuple works")
    check_equal(disconnect.is_tracker_many(iter(urls[:10])), expected[:10], "an iterator works")
    check_equal(disconnect.is_tracker_many(url for url in urls[:10]), expected[:10], "a generator works")
    check_equal(disconnect.is_tracker_many([]), [], "an empty batch is an empty list")


def test_threaded(urls):
    """The matching releases the GIL, so it must be safe from several threads."""
    expected = [disconnect.is_tracker(url) for url in urls]
    results = [None] * 8
    def worker(index):
        results[index] = disconnect.is_tracker_many(urls)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    check(all(result == expected for result in results), "eight threads agree with the single-threaded answer")


def test_list_introspection(reference):
    info = disconnect.list_info()
    check_equal(sorted(info), ["categories", "hosts", "organizations", "rules", "source"], "list_info fields")
    check("disconnect_services.json" in info["source"], "list info names its source")
    check_equal(info["hosts"], len(set(reference.hosts) | set(reference.paths)), "list info host count")
    check_equal(info["categories"], len(reference.categories), "list info category count")

    check_equal(disconnect.categories(), reference.categories, "categories() matches the list, in list order")

    entries = disconnect.entries()
    check_equal(len(entries), info["rules"], "entries() returns every rule")
    check_equal(sorted(entries[0]), ["category", "organization", "organization_url", "pattern"], "entry fields")
    path_rules = sum(len({path for path, _, _ in rules}) for rules in reference.paths.values())
    check_equal(len({e["pattern"] for e in entries}), len(reference.hosts) + path_rules,
                "entries() covers every pattern")
    check(all(e["category"] in reference.categories for e in entries), "every entry has a known category")


def test_runtime_load():
    """load() swaps the list process-wide; it must be reset before anything else runs."""
    custom = {
        "license": "test fixture",
        "categories": {
            "Advertising": [{"Example Org": {"https://example.org/": ["tracker.test", "deep.example.test/beacon"]}}],
            "Analytics": [{"Other Org": {"https://other.test/": ["metrics.test"]}}],
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(custom, handle)
        path = handle.name
    try:
        info = disconnect.load(path)
        check_equal(info["hosts"], 3, "custom list has three hosts")
        check_equal(info["categories"], 2, "custom list has two categories")
        check_equal(info["source"], path, "list info names the loaded path")
        check(disconnect.is_tracker("https://sub.tracker.test/x") is True, "custom list matches")
        check(disconnect.is_tracker("https://doubleclick.net/") is False, "the embedded list is no longer active")
        check_equal(disconnect.tracker_owner("https://metrics.test/"), "Other Org", "custom owner")
        check(disconnect.is_tracker("https://deep.example.test/beacon/1") is True, "custom path rule matches")
        check(disconnect.is_tracker("https://deep.example.test/other") is False, "custom path rule is scoped")
        check_equal(disconnect.categories(), ["Advertising", "Analytics"], "categories() follows the loaded list")
        # A filter that named a category of the embedded list must now fail.
        check_raises(ValueError, lambda: disconnect.is_tracker("https://tracker.test/", "Cryptomining"),
                     "category filters follow the loaded list")

        check_raises(ValueError, lambda: disconnect.load(os.path.join(HERE, "test_module.py")),
                     "loading a non-JSON file fails")
        check_raises(OSError, lambda: disconnect.load("/no/such/list.json"), "loading a missing file fails")
        check_raises(OSError, lambda: disconnect.load(HERE), "loading a directory fails")
        # A failed load leaves the previously loaded list in place.
        check(disconnect.is_tracker("https://tracker.test/") is True, "failed loads do not clear state")
        # load() accepts anything os.PathLike, like open() does.
        check(disconnect.load(pathlib.Path(path))["hosts"] == 3, "load() accepts a path object")
    finally:
        os.unlink(path)

    info = disconnect.reset()
    check(info["hosts"] > 4000, "reset restores the embedded list")
    check(disconnect.is_tracker("https://doubleclick.net/") is True, "embedded list is active again")


def main():
    reference = ReferenceMatcher(LIST_JSON)
    urls = probe_urls(reference)

    test_basics()
    test_null_propagates()
    test_keyword_arguments()
    test_category_filters()
    test_type_errors()
    test_against_reference(reference, urls)
    test_match_agrees_with_the_scalars(urls)
    test_index_points_into_entries(urls)
    test_filtered_against_reference(reference, urls)
    test_batch_matches_the_scalar(urls)
    test_threaded(urls)
    test_list_introspection(reference)
    # Last: this one replaces the process-wide list.
    test_runtime_load()

    if failures:
        print("FAILED (%d of %d checks)" % (len(failures), checks))
        for failure in failures:
            print("  - %s" % failure)
        return 1
    print("all %d module checks passed" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
