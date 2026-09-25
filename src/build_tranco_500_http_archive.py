"""Extract what HTTP Archive says about the crawled top-500 domains.

The paired crawl loaded 500 pages twice on one afternoon, and every bound in
`tests/top500.py` divides by something that crawl measured. This gives those
bounds a denominator that comes from somewhere else entirely: HTTP Archive's
own crawl of the same domains, a different browser on a different day, with
no connection to ours beyond the domain name.

It reads `data/http-archive-urls-50pct`, the 50%-sampled request export, and
writes one row per crawled domain to `data/tranco_500_http_archive.csv`:

    domain, n_req, bytes, tracker_bytes, n_tracker_req, n_tracker_hosts

The 50% export rather than the 1% ones because this needs *composition*: at
one request in a hundred a page keeps two or three of its ~300 requests and
its tracker share is unmeasurable, while at one in two it keeps most of them.
The 1% exports are the right instrument for per-request ratios and the wrong
one for per-page sums; see `llm-classifier/scripts/fit_followups.py`, which
uses both for opposite purposes.

Counts are of *sampled* requests, so they are about half of what the page
really made. Nothing downstream reads them as absolute; they are there to
weight and to sanity-check. `bytes` and `tracker_bytes` are likewise sampled
halves, and the ratio between them -- which is what the bound actually uses
-- is unbiased.

    python src/build_tranco_500_http_archive.py

Takes about six minutes over 222 GB, so the result is committed: it is 350
rows about a fixed list of domains, and re-deriving it is not something a
test run should do. Re-run it when the export is refreshed, and expect the
recorded ratio in `tests/top500.py` to move a little when you do -- HTTP
Archive re-crawls monthly and pages change.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "duckdb-disconnect" / "build" / "disconnect.duckdb_extension"
EXPORT = ROOT / "data" / "http-archive-urls-50pct" / "*.parquet"
CATEGORIES = ROOT / "data" / "tranco_500_categories.csv"
OUT = ROOT / "data" / "tranco_500_http_archive.csv"

COLUMNS = ("domain", "n_req", "bytes", "tracker_bytes", "n_tracker_req",
           "n_tracker_hosts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", default=str(EXPORT))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import duckdb

    if not list(Path(args.export).parent.glob("*.parquet")):
        print(f"{args.export} matches nothing; the 50% export is gitignored "
              f"and has to be downloaded first")
        return 1

    with open(CATEGORIES, newline="") as f:
        domains = [r["domain"] for r in csv.DictReader(f)]

    con = duckdb.connect(config={"allow_unsigned_extensions": "true",
                                 "memory_limit": "12GB"})
    con.sql(f"LOAD '{EXTENSION}'")
    con.sql("create table labelled(domain varchar)")
    con.executemany("insert into labelled values (?)", [(d,) for d in domains])
    # `page_domain` carries the host as crawled, so a domain can appear with
    # and without the www label; both are the same site for our purposes and
    # are summed together.
    rows = con.sql(f"""
        select l.domain,
               count(*) n_req,
               sum(r.transfer_bytes) bytes,
               sum(case when is_tracker(r.url) then r.transfer_bytes else 0 end)
                   tracker_bytes,
               sum(case when is_tracker(r.url) then 1 else 0 end) n_tracker_req,
               count(distinct case when is_tracker(r.url)
                     then disconnect_url_host(r.url) end) n_tracker_hosts
        from read_parquet('{args.export}') r
        join labelled l
          on r.page_domain = l.domain or r.page_domain = 'www.' || l.domain
        where r.transfer_bytes is not null
        group by 1 order by 1
    """).fetchall()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    total = sum(r[2] for r in rows)
    tracker = sum(r[3] for r in rows)
    print(f"{len(rows)} of {len(domains)} crawled domains appear in the export")
    print(f"  {sum(r[1] for r in rows):,} sampled requests, "
          f"{total/1e9:.2f} GB, {tracker/1e9:.2f} GB from trackers "
          f"({tracker/total:.1%})")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
