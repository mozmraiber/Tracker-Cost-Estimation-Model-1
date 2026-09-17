# `disconnect` — a DuckDB extension for tracker classification

A C++ DuckDB extension that answers "is this URL a tracker?" against the
[Disconnect services list](https://github.com/disconnectme/disconnect-tracking-protection),
the list Firefox uses for Enhanced Tracking Protection.

```sql
LOAD 'disconnect.duckdb_extension';

SELECT url, is_tracker(url), tracker_category(url), tracker_owner(url)
FROM requests;
```

```
┌────────────────────────────────────────────────┬────────────┬──────────────────┬──────────────┐
│ url                                            │ is_tracker │ tracker_category │ tracker_owner│
├────────────────────────────────────────────────┼────────────┼──────────────────┼──────────────┤
│ https://www.google-analytics.com/collect?v=1   │ true       │ Analytics        │ Google       │
│ https://cdn.example.org/app.js                 │ false      │ NULL             │ NULL         │
│ https://connect.facebook.net/en_US/fbevents.js │ true       │ Social           │ Meta         │
└────────────────────────────────────────────────┴────────────┴──────────────────┴──────────────┘
```

The list is compiled into the extension, so nothing needs to be downloaded or
joined at query time. Classification runs at roughly 90M URLs/s on an M-series
laptop (10M rows, 12 threads), which is fast enough to leave `is_tracker(url)`
in a `WHERE` clause over a full crawl.

The same matcher is also available as a plain [Python module](#python-module),
for classifying URLs outside SQL.

## Functions

| Function | Returns | Description |
| --- | --- | --- |
| `is_tracker(url)` | `BOOLEAN` | Is the URL on the list, in any category except `Content`? |
| `is_tracker(url, categories)` | `BOOLEAN` | Same, restricted to a comma-separated category list; `Content` counts if named |
| `tracker_category(url)` | `VARCHAR` | The primary category of the match, `NULL` if unlisted |
| `tracker_categories(url)` | `VARCHAR[]` | Every category the match is listed under, `NULL` if unlisted |
| `tracker_owner(url)` | `VARCHAR` | The organization that owns the matched domain |
| `tracker_owner_url(url)` | `VARCHAR` | That organization's home page |
| `tracker_pattern(url)` | `VARCHAR` | The list entry that matched, e.g. `doubleclick.net` |
| `disconnect_url_host(url)` | `VARCHAR` | The host, parsed exactly as the matcher parses it |
| `disconnect_entries()` | table | The whole list: `pattern, category, organization, organization_url` |
| `disconnect_list_info()` | table | Which list is loaded and how large it is |
| `disconnect_load(path)` | table | Replace the list at runtime from a services JSON |
| `disconnect_reset()` | table | Go back to the list embedded at build time |

`NULL` input gives `NULL` output. A URL that is not on the list is `false` for
`is_tracker` and `NULL` for everything else. A URL listed only under `Content`
is also `false` for `is_tracker`, but the other functions still describe it —
see [Categories](#categories).

## Matching rules

* **Input** may be a full URL, a scheme-relative URL, or a bare hostname —
  `NET.HOST`-style columns can be passed straight in. Userinfo, ports, trailing
  root dots and IPv6 brackets are stripped, and matching is case-insensitive.
* **Parent domains count.** `stats.g.doubleclick.net` matches the entry
  `doubleclick.net`. Matching is right-anchored on label boundaries, so
  `notdoubleclick.net` and `doubleclick.net.example.com` do not match.
* **Path-scoped entries** are honoured. Seven entries in the list scope a rule
  to a path (`google.com/pagead/1p-user-list`); those match only when the URL
  carries that path prefix. Passing a bare hostname can therefore never match
  a path rule.
* **The most specific rule wins**: a path rule on a host is preferred over a
  host rule on one of its parents.

## Categories

The list has eleven categories: `Advertising`, `Analytics`, `Social`,
`Cryptomining`, `FingerprintingInvasive`, `FingerprintingGeneral`, `Content`,
`Email`, `EmailAggressive`, `Anti-fraud`, `ConsentManagers`.

`is_tracker(url)` with one argument is true for any category **except
`Content`**. That category lists the first-party properties of tracking
companies — `google.com`, `yandex.ru` and `fonts.googleapis.com` are all on it
— and a browser does not block those, so counting them made the unfiltered
form answer a question nobody was asking:

```sql
SELECT is_tracker('https://www.google.com/search?q=x');           -- false
SELECT is_tracker('https://www.google.com/search?q=x', 'Content'); -- true
SELECT tracker_category('https://www.google.com/search?q=x');      -- 'Content'
```

Only the verdict changes. `tracker_category()`, `tracker_categories()`,
`tracker_owner()`, `tracker_owner_url()` and `tracker_pattern()` all still
look a URL up under every category, so a `Content` host is still described in
full — and naming `Content` in the two-argument form still matches it. A host
listed under `Content` *and* a blocking category is a tracker either way.

Note that this is still broader than what a browser blocks. For something
closer to Firefox's default protection, name the categories explicitly:

```sql
-- Firefox ETP "standard"-ish
SELECT is_tracker(url, 'Advertising,Analytics,Social,Cryptomining') FROM requests;

-- Add fingerprinting, as ETP "strict" does
SELECT is_tracker(url, 'Advertising,Analytics,Social,Cryptomining,'
                    || 'FingerprintingInvasive,FingerprintingGeneral') FROM requests;
```

Category names are matched case-insensitively and an unknown name is an error,
so a typo fails the query rather than silently matching nothing. When the
filter is a constant it is resolved once per query, not per row.

Many hosts are listed under several categories — the fingerprinting lists
overlap heavily with `Advertising` and `Analytics`. `tracker_category()` picks
one, preferring what a host *does* over how it does it (see
`CATEGORY_PRIORITY` in [src/disconnect_list.cpp](src/disconnect_list.cpp));
`tracker_categories()` returns all of them.

## Using a different or newer list

The embedded list is generated from
[`../data/external/disconnect_services.json`](../data/external/disconnect_services.json).
To classify against a different revision without rebuilding:

```sql
SELECT * FROM disconnect_load('data/external/disconnect_services.json');
SELECT * FROM disconnect_reset();  -- back to the embedded list
```

`disconnect_load` reads through DuckDB's file system, so paths that `httpfs`
handles (`s3://`, `https://`) work too. The loaded list is **process-wide**:
it applies to every connection in the process until reset.

To bake a newer list into the binary instead:

```sh
make regen-list   # rewrites src/generated/disconnect_data.cpp
make
```

## Python module

The matcher is also a CPython extension module, with no DuckDB involved:

```python
import disconnect

disconnect.is_tracker("https://www.google-analytics.com/collect?v=1")  # True
disconnect.tracker_category("connect.facebook.net")                    # 'Social'
disconnect.match("https://stats.g.doubleclick.net/x")
# {'pattern': 'doubleclick.net', 'category': 'Advertising',
#  'categories': ['Email', 'Advertising', 'FingerprintingGeneral'],
#  'owner': 'Google', 'owner_url': 'http://www.google.com/'}
```

```sh
pip install ./duckdb-disconnect   # or `make module`, see Building
```

It is built from the same sources as the extension — the list model, the
matcher, the URL parser and the embedded list — so every matching rule and
category note above applies unchanged. Only the calling convention differs:
`None` stands in for SQL `NULL` both in and out, and `ValueError` for what SQL
raises as an error.

| Python | SQL |
| --- | --- |
| `is_tracker(url, categories=None)` | `is_tracker(url[, categories])` |
| `is_tracker_many(urls, categories=None)` | — |
| `tracker_category(url)` | `tracker_category(url)` |
| `tracker_categories(url)` | `tracker_categories(url)` |
| `tracker_owner(url)` | `tracker_owner(url)` |
| `tracker_owner_url(url)` | `tracker_owner_url(url)` |
| `tracker_pattern(url)` | `tracker_pattern(url)` |
| `tracker_index(url)` | — |
| `match(url, categories=None)` | — |
| `url_host(url)` | `disconnect_url_host(url)` |
| `entries()` | `disconnect_entries()` |
| `categories()` | — |
| `list_info()` | `disconnect_list_info()` |
| `load(path)` | `disconnect_load(path)` |
| `reset()` | `disconnect_reset()` |

`categories` takes either the comma-separated string SQL takes or a sequence of
names, and an unknown name is a `ValueError` whether or not there is anything
to classify. `load()` is the same process-wide swap as `disconnect_load()`, but
reads local paths only — there is no DuckDB here to bring `httpfs` along.

Four of these have no SQL counterpart. `categories()`, `match()` and
`tracker_index()` are conveniences: `match()` returns pattern, category,
categories, owner and owner_url from a single lookup instead of five, and
`tracker_index()` returns where the matched entry sits in `entries()`, so a
classification can be joined back onto the list without a second lookup by
pattern — a host listed under several categories has one entry per category,
and the index names the primary one, the entry the other `tracker_*()`
functions describe. `is_tracker_many()` is what makes the module usable at
scale, because per-call overhead otherwise dominates the matching; it resolves
the category filter once and releases the GIL while matching, so threads scale:

```
is_tracker(url) in a list comprehension     7.4M URLs/s
is_tracker_many(urls)                      11.0M URLs/s
is_tracker_many(urls), 8 threads           54.9M URLs/s
```

(470k URLs on an M-series laptop. Still an order of magnitude short of the
extension's 90M URLs/s, which pays no per-URL Python cost at all — for whole
tables, prefer SQL.)

## Building

Requires the Homebrew DuckDB (`brew install duckdb`) — it ships the C++ headers
and the CLI this extension is built and tested against — plus a C++17 compiler
and Python 3 for the two build scripts.

```sh
make            # -> build/disconnect.duckdb_extension and build/python/disconnect*.so
make module     # just the Python module
make test       # every suite
```

Only the extension itself needs DuckDB. The shared sources compile against
`src/include/disconnect_compat.hpp` instead of `duckdb.hpp` when
`DISCONNECT_NO_DUCKDB` is defined, so `make module` and `pip install` need
nothing but a C++17 compiler and the Python development headers. `make module`
builds against whichever interpreter `TEST_PYTHON` names, the project
virtualenv by default.

The build links no DuckDB library of its own; symbols resolve against whichever
DuckDB process loads the extension. Set `DUCKDB_PREFIX` to build against a
different installation:

```sh
make DUCKDB_PREFIX=/path/to/duckdb
```

An extension binary is tied to the DuckDB version and platform recorded in its
metadata footer (`v1.5.5` / `osx_arm64` here) and will refuse to load into any
other. Rebuild after upgrading DuckDB.

## Loading

The binary is unsigned, so DuckDB has to be told to accept it:

```sh
duckdb -unsigned -c "LOAD '$PWD/build/disconnect.duckdb_extension'; SELECT is_tracker('doubleclick.net');"
```

```python
import duckdb
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'build/disconnect.duckdb_extension'")
con.sql("SELECT is_tracker('https://www.google-analytics.com/collect')").show()
```

## Tests

`make test` runs three suites:

* [test/disconnect.test.sql](test/disconnect.test.sql) — SQL-level behaviour,
  each check raising an error if it fails.
* [test/test_disconnect.py](test/test_disconnect.py) — unit checks plus a
  cross-check of ~17.5k generated URLs against an independent reference
  matcher, covering `is_tracker`, `tracker_category`, `tracker_owner`,
  `tracker_pattern` and `tracker_categories`, single- and multi-threaded.
* [test/test_module.py](test/test_module.py) — the Python module over the same
  17.5k URLs, plus its `None` handling, category filters, error types,
  keyword arguments, batching across chunk boundaries, thread safety and
  `load`/`reset`.

The reference matcher in
[test/reference_matcher.py](test/reference_matcher.py) is written straight
against the services JSON and shares no code with the C++ matcher; both suites
check against it.

## Layout

```
src/disconnect_extension.cpp   function registration and the SQL-facing layer
src/python_module.cpp          the same, for CPython
src/disconnect_list.cpp        list model, hash index and matching
src/url_host.cpp               URL -> (host, path) splitting
src/mini_json.cpp              JSON reader used by disconnect_load()
src/include/disconnect_compat.hpp   the handful of duckdb.hpp names the
                               shared sources use, or std equivalents
src/generated/                 the embedded list (generated, committed)
scripts/generate_list_data.py  regenerates the embedded list from the JSON
scripts/append_extension_metadata.py   from duckdb/extension-ci-tools (MIT)
setup.py, pyproject.toml       the pip build of the Python module
```

## Limitations

* No public-suffix awareness: matching is purely by label suffix, as the
  Disconnect list itself is. There is no first-party/third-party
  determination — compare `tracker_owner(request_url)` with
  `tracker_owner(page_url)`, or the hosts themselves, to build one.
* The list is a snapshot. Domains that trackers add after the list revision was
  taken are not matched.

## Licensing

The extension code is part of this repository. The **data** it embeds is
derived from the Disconnect services list, copyright Disconnect, Inc., licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). That
license carries a non-commercial restriction and travels with
`src/generated/disconnect_data.cpp` and with any binary built from it —
the extension, the Python module and any wheel of it alike.
