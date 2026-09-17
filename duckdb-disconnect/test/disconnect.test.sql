-- SQL test suite for the `disconnect` extension.
-- Run with: make test-sql
-- Every check raises an error when it fails, so the CLI exits non-zero.

LOAD 'build/disconnect.duckdb_extension';

CREATE OR REPLACE MACRO must(condition, message) AS
    CASE WHEN condition THEN 'ok' ELSE error('FAILED: ' || message) END;

SELECT 'basic classification' AS suite, min(assertion) AS status FROM (VALUES
    (must(is_tracker('https://www.google-analytics.com/collect?v=1'), 'google-analytics is a tracker')),
    (must(is_tracker('doubleclick.net'), 'bare hostname is accepted')),
    (must(NOT is_tracker('https://example.com/index.html'), 'example.com is not a tracker')),
    (must(is_tracker(NULL) IS NULL, 'NULL in, NULL out')),
    (must(NOT is_tracker(''), 'empty string is not a tracker'))
) t(assertion);

SELECT 'parent domain walk' AS suite, min(assertion) AS status FROM (VALUES
    (must(is_tracker('https://stats.g.doubleclick.net/j/collect'), 'subdomain matches parent')),
    (must(NOT is_tracker('https://notdoubleclick.net/'), 'suffix without a dot does not match')),
    (must(NOT is_tracker('https://doubleclick.net.example.com/'), 'parent walk is right-anchored'))
) t(assertion);

SELECT 'url shapes' AS suite, min(assertion) AS status FROM (VALUES
    (must(is_tracker('//DoubleClick.NET/x'), 'scheme-relative and mixed case')),
    (must(is_tracker('https://doubleclick.net:8443/x'), 'port is ignored')),
    (must(is_tracker('https://user:pw@doubleclick.net/x'), 'credentials are ignored')),
    (must(is_tracker('https://doubleclick.net./x'), 'trailing root dot is ignored')),
    (must(disconnect_url_host('https://user@Example.COM:443/a/b?c#d') = 'Example.COM', 'host extraction')),
    (must(disconnect_url_host('https://[2001:db8::1]:8080/x') = '2001:db8::1', 'ipv6 host extraction'))
) t(assertion);

SELECT 'path rules' AS suite, min(assertion) AS status FROM (VALUES
    (must(is_tracker('https://www.google.com/pagead/1p-user-list/123', 'Advertising'),
          'path rule matches under Advertising')),
    (must(NOT is_tracker('https://www.google.com/search?q=cats', 'Advertising'),
          'a path outside the rule does not match')),
    (must(tracker_pattern('https://www.google.com/pagead/1p-user-list/123') = 'google.com/pagead/1p-user-list',
          'the matched pattern is reported'))
) t(assertion);

SELECT 'categories' AS suite, min(assertion) AS status FROM (VALUES
    (must(tracker_category('https://securepubads.g.doubleclick.net/tag/js/gpt.js') = 'Advertising',
          'a functional category outranks the fingerprinting overlays')),
    (must(list_contains(tracker_categories('https://doubleclick.net/x'), 'FingerprintingGeneral'),
          'every category is reported')),
    (must(tracker_category('https://example.com/') IS NULL, 'no match yields NULL')),
    (must(tracker_categories('https://example.com/') IS NULL, 'no match yields NULL for the list too')),
    (must(is_tracker('https://www.google-analytics.com/collect', 'Analytics'), 'category filter selects')),
    (must(NOT is_tracker('https://www.google-analytics.com/collect', 'Cryptomining'), 'category filter excludes')),
    (must(is_tracker('https://www.google-analytics.com/collect', ' analytics , Social '),
          'filters are trimmed and case-insensitive')),
    -- google.com is listed, but only under Content, which the unfiltered form
    -- of is_tracker() leaves out. The describing functions still report it, so
    -- the classification is not lost — only the verdict changes.
    (must(NOT is_tracker('https://www.google.com/search?q=cats'),
          'Content alone does not make a tracker')),
    (must(is_tracker('https://www.google.com/search?q=cats', 'Content'),
          'naming Content explicitly still matches')),
    (must(tracker_category('https://www.google.com/search?q=cats') = 'Content',
          'tracker_category still reports what the list says')),
    (must(tracker_owner('https://www.google.com/search?q=cats') = 'Google',
          'tracker_owner still reports what the list says')),
    -- A host listed under Content *and* a blocking category is still a tracker.
    (must(is_tracker('https://www.google-analytics.com/collect'),
          'a non-Content category still matches unfiltered'))
) t(assertion);

SELECT 'organizations' AS suite, min(assertion) AS status FROM (VALUES
    (must(tracker_owner('https://connect.facebook.net/en_US/fbevents.js') = 'Meta', 'owner lookup')),
    (must(tracker_owner_url('https://connect.facebook.net/en_US/fbevents.js') = 'https://www.meta.com/',
          'owner url lookup')),
    (must(tracker_owner('https://example.com/') IS NULL, 'no match yields NULL'))
) t(assertion);

SELECT 'list introspection' AS suite, min(assertion) AS status FROM (VALUES
    (must((SELECT count(*) FROM disconnect_entries()) > 4000, 'disconnect_entries returns the whole list')),
    (must((SELECT count(*) FROM disconnect_entries() WHERE pattern = 'doubleclick.net') = 3,
          'doubleclick.net is listed three times')),
    (must((SELECT hosts > 4000 AND categories = 11 AND organizations > 1000 FROM disconnect_list_info()),
          'list info is populated'))
) t(assertion);

-- Joining against a table, the way a real query would use it.
SELECT 'table usage' AS suite, min(assertion) AS status FROM (VALUES
    (must((SELECT count(*) FROM (VALUES
        ('https://www.google-analytics.com/analytics.js'),
        ('https://connect.facebook.net/en_US/sdk.js'),
        ('https://cdn.example.org/app.js')
     ) AS r(url) WHERE is_tracker(r.url)) = 2, 'two of three requests are trackers'))
) t(assertion);

SELECT 'all SQL tests passed' AS result;
