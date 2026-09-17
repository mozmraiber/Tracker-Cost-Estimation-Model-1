#!/usr/bin/env python3
"""Test suite for the `disconnect` DuckDB extension.

Run with: make test-python

Besides the usual unit checks, this cross-checks the extension against an
independent reference matcher written directly against the source JSON, over
every pattern in the list.
"""

import json
import os
import sys
import tempfile

import duckdb

from reference_matcher import LIST_JSON, ReferenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXTENSION = os.path.join(HERE, "..", "build", "disconnect.duckdb_extension")

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def check_equal(actual, expected, message):
    check(actual == expected, "%s (got %r, expected %r)" % (message, actual, expected))


def run(connection, sql, *params):
    return connection.execute(sql, list(params)).fetchall()


def test_basics(con):
    check(run(con, "SELECT is_tracker('https://www.google-analytics.com/collect')")[0][0] is True,
          "google-analytics is a tracker")
    check(run(con, "SELECT is_tracker('https://example.com/')")[0][0] is False, "example.com is not a tracker")
    check(run(con, "SELECT is_tracker(NULL)")[0][0] is None, "NULL propagates")
    check(run(con, "SELECT is_tracker('')")[0][0] is False, "empty string is not a tracker")
    check(run(con, "SELECT tracker_category('https://example.com/')")[0][0] is None, "no category without a match")
    check(run(con, "SELECT tracker_categories('https://example.com/')")[0][0] is None, "no category list either")

    # Odd but not invalid input must not crash or match.
    for url in ["not a url", "http://", "https://:8080/", "///", "@", ":", "..", "http://.../x", "%%%",
                "https://" + "a" * 5000 + ".com/"]:
        run(con, "SELECT is_tracker(?)", url)
    check(True, "odd input is handled")


def test_null_and_mixed_batches(con):
    rows = run(con, """
        SELECT url, is_tracker(url), tracker_category(url), tracker_owner(url)
        FROM (VALUES ('https://doubleclick.net/x'), (NULL), ('https://example.com/'),
                     ('https://www.google-analytics.com/collect')) AS t(url)
        ORDER BY url NULLS LAST
    """)
    # Ordered by URL: doubleclick, example.com, google-analytics, then NULL.
    check_equal([r[1] for r in rows], [True, False, True, None], "mixed batch with NULLs")


def test_category_filters(con):
    check(run(con, "SELECT is_tracker('https://www.google-analytics.com/x', 'Analytics')")[0][0] is True,
          "filter selects")
    check(run(con, "SELECT is_tracker('https://www.google-analytics.com/x', 'Advertising')")[0][0] is False,
          "filter excludes")
    # A non-constant filter cannot be folded at bind time; the per-row path must
    # produce the same answers.
    rows = run(con, """
        SELECT is_tracker('https://www.google-analytics.com/x', f)
        FROM (VALUES ('Analytics'), ('Advertising'), ('Analytics,Advertising')) AS t(f)
    """)
    check_equal([r[0] for r in rows], [True, False, True], "per-row category filter")

    for bad in ["NotACategory", "", "  ", "Analytics,Nope"]:
        try:
            run(con, "SELECT is_tracker('https://doubleclick.net/', ?)", bad)
            check(False, "filter %r should be rejected" % bad)
        except duckdb.Error:
            check(True, "filter %r is rejected" % bad)


def test_against_reference(con, reference):
    """Compare the extension with the reference matcher over the whole list."""
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

    con.execute("CREATE OR REPLACE TABLE probe(url VARCHAR)")
    con.executemany("INSERT INTO probe VALUES (?)", [(url,) for url in urls])
    rows = run(con, """
        SELECT url, is_tracker(url), tracker_category(url), tracker_owner(url), tracker_pattern(url),
               tracker_categories(url)
        FROM probe
    """)
    check_equal(len(rows), len(urls), "every probe URL is answered")

    # is_tracker() is checked against the reference's default categories and the
    # describing functions against every category, because that asymmetry is
    # the point: a Content-only host is described but is not called a tracker.
    mismatches = []
    for url, tracker, category, owner, pattern, category_list in rows:
        expected = reference.match(url)
        if expected is None:
            if tracker or category or owner or pattern or category_list:
                mismatches.append((url, "expected no match", (tracker, category, owner, pattern)))
            continue
        expected_category, expected_org, expected_pattern = expected
        actual = (tracker, category, owner, pattern, category_list)
        wanted = (reference.is_tracker(url), expected_category, expected_org, expected_pattern,
                  reference.categories_of(url))
        if actual != wanted:
            mismatches.append((url, wanted, actual))
    check(not mismatches, "reference mismatches: %s" % mismatches[:5])
    print("  cross-checked %d URLs against the reference matcher" % len(urls))


def test_filtered_against_reference(con, reference):
    allowed = ["Advertising", "Analytics", "Social", "Cryptomining"]
    filter_text = ",".join(allowed)
    rows = run(con, "SELECT url, is_tracker(url, '%s') FROM probe" % filter_text)
    mismatches = [
        (url, tracker) for url, tracker in rows
        if tracker != (reference.match(url, allowed=set(allowed)) is not None)
    ]
    check(not mismatches, "filtered reference mismatches: %s" % mismatches[:5])


def test_parallel_scan(con, reference):
    """The same answers must come back with several threads and many rows."""
    con.execute("SET threads=8")
    listed, total = run(con, """
        SELECT count(*) FILTER (WHERE is_tracker(url)), count(*)
        FROM (SELECT url FROM probe, range(20) r)
    """)[0]
    expected = sum(1 for (url,) in run(con, "SELECT url FROM probe") if reference.is_tracker(url))
    check_equal(listed, expected * 20, "parallel scan agrees with the reference")
    check_equal(total, len(run(con, "SELECT url FROM probe")) * 20, "parallel scan sees every row")
    con.execute("SET threads=4")


def test_entries_table(con, reference):
    rows = run(con, "SELECT count(*), count(DISTINCT pattern), count(DISTINCT organization) FROM disconnect_entries()")
    # A host listed under several categories is one host but several patterns
    # only when the patterns differ, and a path rule is a pattern of its own.
    path_rules = sum(len({path for path, _, _ in rules}) for rules in reference.paths.values())
    expected_patterns = len(reference.hosts) + path_rules
    expected_hosts = len(set(reference.hosts) | set(reference.paths))
    check_equal(rows[0][1], expected_patterns, "disconnect_entries covers every pattern")
    check(rows[0][0] >= expected_patterns, "one row per (pattern, category, organization)")

    info = run(con, "SELECT source, hosts, rules, categories, organizations FROM disconnect_list_info()")[0]
    check_equal(info[1], expected_hosts, "list info host count")
    check_equal(info[3], len(reference.categories), "list info category count")
    check("disconnect_services.json" in info[0], "list info names its source")


def test_runtime_load(con):
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
        info = run(con, "SELECT source, hosts, rules, categories, organizations FROM disconnect_load(?)", path)[0]
        check_equal(info[1], 3, "custom list has three hosts")
        check_equal(info[3], 2, "custom list has two categories")
        check(run(con, "SELECT is_tracker('https://sub.tracker.test/x')")[0][0] is True, "custom list matches")
        check(run(con, "SELECT is_tracker('https://doubleclick.net/')")[0][0] is False,
              "the embedded list is no longer active")
        check_equal(run(con, "SELECT tracker_owner('https://metrics.test/')")[0][0], "Other Org", "custom owner")
        check(run(con, "SELECT is_tracker('https://deep.example.test/beacon/1')")[0][0] is True,
              "custom path rule matches")
        check(run(con, "SELECT is_tracker('https://deep.example.test/other')")[0][0] is False,
              "custom path rule is scoped")
        # A filter that named a category of the embedded list must now fail.
        try:
            run(con, "SELECT is_tracker('https://tracker.test/', 'Cryptomining')")
            check(False, "category filters follow the loaded list")
        except duckdb.Error:
            check(True, "category filters follow the loaded list")

        try:
            run(con, "SELECT * FROM disconnect_load(?)", os.path.join(HERE, "test_disconnect.py"))
            check(False, "loading a non-JSON file fails")
        except duckdb.Error:
            check(True, "loading a non-JSON file fails")
        try:
            run(con, "SELECT * FROM disconnect_load('/no/such/list.json')")
            check(False, "loading a missing file fails")
        except duckdb.Error:
            check(True, "loading a missing file fails")
        # A failed load leaves the previously loaded list in place.
        check(run(con, "SELECT is_tracker('https://tracker.test/')")[0][0] is True, "failed loads do not clear state")
    finally:
        os.unlink(path)

    info = run(con, "SELECT hosts FROM disconnect_reset()")[0]
    check(info[0] > 4000, "reset restores the embedded list")
    check(run(con, "SELECT is_tracker('https://doubleclick.net/')")[0][0] is True, "embedded list is active again")


def main():
    extension = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXTENSION
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute("LOAD '%s'" % os.path.abspath(extension))
    reference = ReferenceMatcher(LIST_JSON)

    test_basics(con)
    test_null_and_mixed_batches(con)
    test_category_filters(con)
    test_against_reference(con, reference)
    test_filtered_against_reference(con, reference)
    test_parallel_scan(con, reference)
    test_entries_table(con, reference)
    test_runtime_load(con)

    if failures:
        print("FAILED (%d of %d checks)" % (len(failures), checks))
        for failure in failures:
            print("  - %s" % failure)
        return 1
    print("all %d Python checks passed" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
