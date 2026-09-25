"""Measure `FOLLOWUP_BYTES_PER_REQUEST` and `FOLLOWUP_CPU_MS_PER_KIB`.

`build_table.py` fits the byte estimate from a log of requests that completed.
That log cannot say anything about the cascade -- the requests a blocked
tracker would have gone on to make are, by construction, not in a log of
requests that were made -- so the two constants in `src/lib.rs` come from
here, and from three datasets whose weaknesses are different:

  1% URL exports  `data/http-archive-urls-1pct{,_new}`, 9.7M tracker requests
                  over 3.2M page domains, sampled uniformly at the *request*
                  level. Nothing in it was blocked, so there is no
                  counterfactual, but it carries `initiator_type`: the request
                  tree's shape. Parser-initiated tracker requests are roots the
                  markup asked for, script-initiated ones were pulled in by
                  something. Huge sample, and every quantity taken from it is
                  a ratio of per-request sums, which uniform thinning leaves
                  unbiased.

  50% URL export  `data/http-archive-urls-50pct`, ~2.3B requests. No
                  `initiator_type`, so it says nothing about the tree -- but
                  it is dense enough that a page keeps most of its requests,
                  which is what makes per-page composition, and therefore a
                  per-*site* fit, possible at all. The 1% exports cannot do
                  this: at one request in a hundred, two requests from the
                  same page almost never both survive. What it cannot do is
                  supply a counterfactual: a coefficient here is what
                  co-occurs with a host, which is the same thing as a cascade
                  only inside the list. See `per_host_cascade_split`.

  Paired crawl    `tests/top500.py`, 271 pages where one arm blocked and the
                  other did not, each loaded ten times in each arm. The only
                  counterfactual anywhere in the repo, and the only population
                  that is exactly what the constant is applied to. Ten passes
                  fixed the part of its noise that comes from a page loading
                  differently twice and left the part that comes from having
                  271 pages, so the whole-page instrument is still too loud to
                  set a constant from: 26.1 KB a request, 95% [-26, 70].

Run:

    python llm-classifier/scripts/fit_followups.py

The 50% export is 222 GB; the three per-page aggregates it needs are
materialised once under `data/cache` (a minute or two each) and reused.
`--rescan-50pct` rebuilds them, `--skip-50pct` leaves that section out.

What the crawl *is* emphatic about is the form rather than the magnitude:
whether a blocked request's subtree scales with its own size or not. It does
not -- see `joint_fit`, `by_request_size` and `grade_form`, which are the
evidence for `FOLLOWUP_BYTES_PER_REQUEST` being per request, and which
re-measure the proportional form it replaced on every run.

Nine refinements are measured here and none is shipped -- a figure per
tracker category, a factor per page category, two per tracker site, a cascade
that scales with the blocked request's own size, saturation in the number of
loaders blocked on a page, a figure of its own for consent managers, and the
unlisted half of the cascade measured directly instead of by subtraction, and
a page-level intercept fitted alongside the per-request term. The last three
are new, and both are answered by a count rather than by a fit: ETP
blocks a consent manager zero times in 18,891 blocked requests, and the
unlisted delta carries a 229% pooled standard error. They are reported every run, because "we tried it
and the data said no" is only worth anything if it re-runs. The per-site pair
is graded twice over, against the whole-page delta the cascade is calibrated
on and against the Disconnect-only delta that is four times quieter, because
a shape too small for the loud instrument to see might still have shown up in
the quiet one. It does not: see `grade_shape`.

The last two are new, and saturation used to be recorded as left open rather
than rejected. Two things changed. Its bucket table was reading ten-pass
counts against single-pass bucket edges, which put 190 of 200 pages in one
bucket and made the table unreadable; and a bucket table cannot settle the
question in any case, because a page-level intercept divided by k falls in k
exactly as a concave count does. `grade_saturation` and `grade_by_size` put
both shapes head to head with the flat form at a fixed total, which is how
`grade_form` settled per request against per byte, and both lose on the quiet
instrument.

One quantity here is not a constant for `src/lib.rs` but an input to the
bounds in `tests/top500.py`: how much of the cascade lands back on hosts the
Disconnect list names, which is what decides how sharp the crawl's quiet
instrument can be. `listed_factor_from_delta` measures it and
`per_host_cascade_split` records why HTTP Archive cannot.

Nothing is regenerated by running this. It prints; the constants are edited
into `src/lib.rs` by hand, the way a two-line calibration should be.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math
import random
import statistics as st
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "duckdb-disconnect" / "build" / "disconnect.duckdb_extension"
#: The uniform request-level exports, which carry `initiator_type`.
URL_EXPORTS = [ROOT / "data" / "http-archive-urls-1pct" / "*.parquet",
               ROOT / "data" / "http-archive-urls-1pct_new" / "*.parquet"]
#: The dense export, which does not.
HA50_GLOB = ROOT / "data" / "http-archive-urls-50pct" / "*.parquet"
#: One page domain in `HA50_MODULUS`, aggregated to (page, tracker host).
#: `md5_number_lower` rather than `hash` for the reason `tests/http_archive.py`
#: gives: it is stable across DuckDB versions, so the sample does not move.
HA50_CACHE = (ROOT / "data" / "cache"
              / "http-archive-urls-50pct-trackerhost-pagemod20.parquet")
#: The same page sample, aggregated to the page: its total bytes, the part of
#: them the Disconnect list names, and the third-party part it does not. What
#: the second cache is for is the *split* -- whether a tracker's cascade lands
#: on hosts the list names, which is what decides whether the crawl's quiet
#: Disconnect-only instrument can see it. See `per_host_cascade_split`.
HA50_MASS_CACHE = (ROOT / "data" / "cache"
                   / "http-archive-urls-50pct-pagemass-pagemod20.parquet")
#: The same page sample again, aggregated to (page, Disconnect category), for
#: the per-category refinement in `per_category_cascade`.
HA50_CATEGORY_CACHE = (ROOT / "data" / "cache"
                       / "http-archive-urls-50pct-category-pagemod20.parquet")
HA50_MODULUS = 20

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import top500  # noqa: E402
from compare_estimate_vs_etp import context_for  # noqa: E402
from compare_tracking_arms import load_har_sizes  # noqa: E402

#: Contexts `FOLLOWUP_BYTES_PER_REQUEST` gives a non-zero figure: the two that
#: run code. The fit pools them and holds everything else at its own bytes.
CASCADING = ("SCRIPT", "HTML")

#: The shipped figure, so the report below can be read against it: bytes of
#: pruned subtree per blocked request in a cascading context.
SHIPPED_BYTES_PER_REQUEST = 47_000

#: What it shipped until 2026-09, as a multiple of the blocked request's own
#: size. Kept because the head-to-head in `form_report` is the evidence for
#: the change, and evidence that cannot be re-run is not evidence.
RETIRED_FACTOR_PER_BYTE = 2.3

#: How `grade_shape` names its two instruments in the report.
INSTRUMENT = {"page": "whole-page delta (loud)",
              "tracking": "listed-only delta (quiet)"}

#: `CPU_MS_PER_KIB` from `src/lib.rs`, to weight by the descendant mix. Kept
#: as a copy rather than parsed out of the Rust, with the same discipline as
#: `build_table.py`'s mirrored enums: if one moves, this must be moved too.
CPU_MS_PER_KIB = {
    "script": 2.0, "image": 0.10, "video": 0.02, "other": 0.10, "json": 0.10,
    "audio": 0.02, "css": 0.30, "font": 0.05, "html": 0.30, "text": 0.20,
    "wasm": 2.00, "xml": 0.20,
}


def _ctx(resource_type: str) -> str:
    return str(context_for(resource_type)).rsplit(".", 1)[-1]


def _con():
    """A DuckDB connection with `is_tracker` available."""
    import duckdb

    con = duckdb.connect(config={"allow_unsigned_extensions": "true",
                                 "memory_limit": "12GB"})
    con.sql(f"LOAD '{EXTENSION}'")
    return con


def _present(globs: list[Path]) -> list[str]:
    return [str(g) for g in globs if list(g.parent.glob("*.parquet"))]


# --------------------------------------------------------------------------- #
# HTTP Archive, 1% uniform: the tree's shape
# --------------------------------------------------------------------------- #
def tree_shape() -> dict | None:
    """Root/descendant split over every tracker request in the 1% exports.

    Read this as an upper bound on the cascade, not as a measurement of it.
    A script-initiated tracker request is one that some script asked for, and
    the export does not say *which* script. If the parent is another tracker,
    ETP never issues the child and the child is cascade; if the parent is the
    page's own code, ETP refuses the child directly and it belongs in the
    denominator instead. The split between those two cases is exactly what is
    missing, and it is why this brackets the crawl's number rather than
    replacing it.
    """
    globs = _present(URL_EXPORTS)
    if not globs:
        return None
    con = _con()
    lst = ", ".join(f"'{g}'" for g in globs)
    con.sql(f"create view t as select * from read_parquet([{lst}]) "
            f"where is_tracker(url) and transfer_bytes is not null")
    n, pages, total, root, desc = con.sql("""
        select count(*), count(distinct page_domain), sum(transfer_bytes),
               sum(case when initiator_type = 'parser' then transfer_bytes else 0 end),
               sum(case when initiator_type = 'script' then transfer_bytes else 0 end)
        from t""").fetchone()
    mix = con.sql("""
        select resource_type,
               sum(transfer_bytes) * 1.0 / sum(sum(transfer_bytes)) over ()
        from t where initiator_type = 'script' group by 1 order by 2 desc
    """).fetchall()
    kinds = tuple(c.lower() for c in CASCADING)
    casc_root, casc_root_n, root_n = con.sql(f"""
        select sum(case when initiator_type = 'parser'
                         and resource_type in {kinds}
                        then transfer_bytes else 0 end),
               count(case when initiator_type = 'parser'
                           and resource_type in {kinds} then 1 end),
               count(case when initiator_type = 'parser' then 1 end)
        from t""").fetchone()
    # The array ships bytes per request, so the same descendant mass is also
    # divided by how many roots there were rather than by how many bytes they
    # held. Counted, not converted from the ratios above: those two have
    # different denominators and multiplying either by the other's mean size
    # would mix them.
    return dict(n=n, pages=pages, total=total, root=root, desc=desc,
                casc_root=casc_root, casc_root_n=casc_root_n, root_n=root_n,
                mix=mix, factor=desc / root, factor_casc=desc / casc_root,
                per_root_request=desc / root_n,
                per_casc_root_request=desc / casc_root_n,
                cpu_ms_per_kib=sum(share * CPU_MS_PER_KIB.get(t, 0.10)
                                   for t, share in mix))


def per_host_parser_share(min_requests: int = 20) -> dict[str, float]:
    """Per tracker host, the share of its bytes the markup asks for directly.

    The per-site signal the 1% exports can offer: a host usually written into
    the page is a loader, and a loader is what has a subtree to lose. Taken
    over the uniform exports rather than `per_request_1pct.csv`, whose hosts
    are a deliberately skewed sample (`sql/04_per_request_features.sql` keeps
    the 200 highest-variance and 50 most common tracker domains) and which for
    that reason disagrees with this by a factor of two.
    """
    globs = _present(URL_EXPORTS)
    if not globs:
        return {}
    lst = ", ".join(f"'{g}'" for g in globs)
    return {h: s for h, s in _con().sql(f"""
        select disconnect_url_host(url),
               sum(case when initiator_type = 'parser' then transfer_bytes else 0 end)
                 * 1.0 / nullif(sum(transfer_bytes), 0)
        from read_parquet([{lst}])
        where is_tracker(url) and transfer_bytes is not null
        group by 1 having count(*) >= {min_requests}
    """).fetchall() if s is not None}


# --------------------------------------------------------------------------- #
# HTTP Archive, 50%: per-page composition, and a per-site fit
# --------------------------------------------------------------------------- #
def build_ha50_cache() -> None:
    """Materialise (page, tracker host) -> bytes for 1 page domain in 20."""
    if not list(HA50_GLOB.parent.glob("*.parquet")):
        raise FileNotFoundError(HA50_GLOB.parent)
    HA50_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _con().sql(f"""
        COPY (
            select page_domain, disconnect_url_host(url) host,
                   sum(transfer_bytes) bytes, count(*) n
            from read_parquet('{HA50_GLOB}')
            where is_tracker(url) and transfer_bytes is not null
              and md5_number_lower(page_domain) % {HA50_MODULUS} = 0
            group by 1, 2
        ) TO '{HA50_CACHE}' (FORMAT parquet, COMPRESSION zstd)
    """)


def build_ha50_mass_cache() -> None:
    """Materialise each sampled page's bytes, split by who served them.

    Three columns, and the middle one is the whole point: `tracker_bytes` is
    what the Disconnect-only instrument in `tests/top500.py` can see and
    `tp_other_bytes` is the third-party mass it cannot. Same page sample as
    `build_ha50_cache`, so the two join on `page_domain`.

    First-party bytes are left out of both: a blocked tracker does not prune
    the page's own assets, so they are neither cascade nor a denominator for
    it. "Third party" is decided by comparing registrable-ish suffixes after
    dropping a `www.` -- crude around multi-label public suffixes, which is
    tolerable because this feeds a share rather than a per-host number.
    """
    if not list(HA50_GLOB.parent.glob("*.parquet")):
        raise FileNotFoundError(HA50_GLOB.parent)
    HA50_MASS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _con().sql(rf"""
        COPY (
            with r as (
                select page_domain,
                       regexp_replace(disconnect_url_host(url), '^www\.', '') host,
                       regexp_replace(page_domain, '^www\.', '') page,
                       is_tracker(url) trk, transfer_bytes b
                from read_parquet('{HA50_GLOB}')
                where transfer_bytes is not null
                  and md5_number_lower(page_domain) % {HA50_MODULUS} = 0
            )
            select page_domain,
                   sum(b) total_bytes,
                   sum(case when trk then b else 0 end) tracker_bytes,
                   sum(case when not trk and host <> page
                              and not host like '%.' || page
                              and not page like '%.' || host
                            then b else 0 end) tp_other_bytes,
                   count(*) n
            from r group by 1
        ) TO '{HA50_MASS_CACHE}' (FORMAT parquet, COMPRESSION zstd)
    """)


def _ha50_presence(min_pages: int) -> dict:
    """The design the two fits below share: which tracker hosts each page has.

    Returns the host list, the presence matrix, the per-page tracker mass, the
    bytes of hosts below the cut, and each host's own mean bytes and requests
    per page. Kept in one place so the listed and unlisted fits cannot end up
    describing different populations.
    """
    import numpy as np
    import scipy.sparse as sp

    con = _con()
    con.sql(f"create view ph as select * from '{HA50_CACHE}'")
    hosts = [h for (h,) in con.sql(f"""select host from ph group by host
             having count(*) >= {min_pages} order by count(*) desc""").fetchall()]
    idx = {h: i for i, h in enumerate(hosts)}
    raw = con.sql("select page_domain, host, bytes, n from ph").df()

    # One page in a thousand carries tens of MB of tracker video; left alone
    # it would set every coefficient.
    cap = np.quantile(raw.bytes, 0.999)
    raw["b"] = np.minimum(raw.bytes, cap) / 1024.0
    pages = raw.groupby("page_domain", sort=False)["b"].sum()
    pidx = {d: i for i, d in enumerate(pages.index)}
    y = pages.to_numpy(float)
    rows = raw.page_domain.map(pidx).to_numpy()
    hid = raw.host.map(idx)
    inside = hid.notna().to_numpy()

    k = len(hosts)
    x = sp.csr_matrix((np.ones(inside.sum()),
                       (rows[inside], hid[inside].astype(int).to_numpy())),
                      shape=(len(y), k))
    x.data[:] = 1.0
    other = np.zeros(len(y))
    np.add.at(other, rows[~inside], raw.b.to_numpy()[~inside])
    return dict(hosts=hosts, idx=idx, x=x, other=other, y_listed=y,
                page_index=pages.index,
                agg=raw[inside].groupby("host").agg(own=("b", "mean"),
                                                    reqs=("n", "mean")))


def _nnls_fit(design, y):
    """Non-negative least squares through the normal equations."""
    import numpy as np
    from scipy.optimize import nnls

    gram = (design.T @ design).toarray()
    chol = np.linalg.cholesky(gram + 1e-6 * np.eye(design.shape[1]))
    beta, _ = nnls(chol.T, np.linalg.solve(chol, design.T @ y), maxiter=40000)
    return beta


def build_ha50_category_cache() -> None:
    """Materialise (page, Disconnect category) -> bytes, requests."""
    if not list(HA50_GLOB.parent.glob("*.parquet")):
        raise FileNotFoundError(HA50_GLOB.parent)
    HA50_CATEGORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _con().sql(f"""
        COPY (
            select page_domain, tracker_category(url) cat,
                   sum(transfer_bytes) bytes, count(*) n
            from read_parquet('{HA50_GLOB}')
            where is_tracker(url) and transfer_bytes is not null
              and md5_number_lower(page_domain) % {HA50_MODULUS} = 0
            group by 1, 2
        ) TO '{HA50_CATEGORY_CACHE}' (FORMAT parquet, COMPRESSION zstd)
    """)


def category_shape_from_ha50(min_requests: int = 50_000) -> tuple[dict, str]:
    """Per-category cascade from the 50% export, which does not survive either.

    Same idea as `per_host_cascade_bytes` with the Disconnect category in place
    of the host, on the theory that six coefficients over 510k pages must be
    better identified than 1,826 were. They are better identified and they are
    identified on the wrong thing: Analytics is present on almost every page,
    so its column behaves as the intercept and collects the page-level
    baseline, leaving Advertising -- 7.5M requests, the densest category in the
    export -- with a fitted 1.4 KB a request against its own 5.0 KB mean, and
    therefore no cascade at all. The crawl says the reverse. Reported so the
    next person reaching for this sees where it lands.
    """
    import numpy as np
    import scipy.sparse as sp

    con = _con()
    raw = con.sql(f"select page_domain, cat, bytes, n from "
                  f"'{HA50_CATEGORY_CACHE}'").df()
    raw["cat"] = raw["cat"].fillna("unlisted").astype(str)
    counts = raw.groupby("cat")["n"].sum()
    means = raw.groupby("cat")["bytes"].sum() / counts
    cats = [c for c in counts.sort_values(ascending=False).index
            if counts[c] >= min_requests]
    idx = {c: i for i, c in enumerate(cats)}
    cap = np.quantile(raw.bytes, 0.999)
    raw["b"] = np.minimum(raw.bytes, cap)
    pages = raw.groupby("page_domain", sort=False)["b"].sum()
    pidx = {d: i for i, d in enumerate(pages.index)}
    y = pages.to_numpy(float)
    rows = raw.page_domain.map(pidx).to_numpy()
    col = raw.cat.map(idx)
    inside = col.notna().to_numpy()
    x = sp.csr_matrix((raw.n.to_numpy(float)[inside],
                       (rows[inside], col[inside].astype(int).to_numpy())),
                      shape=(len(y), len(cats)))
    other = np.zeros(len(y))
    np.add.at(other, rows[~inside], raw.n.to_numpy(float)[~inside])
    gamma = _nnls_fit(sp.hstack([x, sp.csr_matrix(other.reshape(-1, 1))]).tocsr(), y)
    shape = {c: max(gamma[idx[c]] - means[c], 0.0) for c in cats}
    hot = max(shape, key=shape.get)
    note = (f"{len(cats)} categories over {len(y):,} pages; all of the cascade "
            f"lands on {hot}, which is the column that doubles as the "
            f"intercept, and none on Advertising")
    return shape, note


def per_host_cascade_bytes(min_pages: int = 50) -> tuple[dict[str, float], str]:
    """Fit each tracker host's additive contribution to a page's tracker mass.

    The model is the one the estimator implies: a page's total tracker bytes
    are the sum of what each tracker on it brings, so

        T_p = sum_h present(h, p) * beta_h + (bytes of hosts below the cut)

    fitted by non-negative least squares over the sampled pages. Fitting the
    hosts *jointly* is the point -- it is what stops a shared ad stack being
    credited in full to every tracker that co-occurs with it, which is the
    flaw in reading `E[page bytes | host present]` straight off the data.

    `beta_h` less the host's own mean bytes is what it brings *besides
    itself*, i.e. its cascade, and dividing by its mean requests per page puts
    that on the per-request footing `estimate_resources` works in.

    Returns (per-request cascade bytes by host, a one-line description).
    """
    import scipy.sparse as sp

    p = _ha50_presence(min_pages)
    k = len(p["hosts"])
    design = sp.hstack([p["x"],
                        sp.csr_matrix(p["other"].reshape(-1, 1))]).tocsr()
    beta = _nnls_fit(design, p["y_listed"])

    out = {}
    for h, i in p["idx"].items():
        a = p["agg"].loc[h]
        out[h] = max(beta[i] - a.own, 0.0) / max(a.reqs, 1.0) * 1024.0
    note = (f"{k} hosts over {len(p['y_listed']):,} pages; the coefficient on "
            f"the bytes of hosts below the cut comes out at {beta[k]:.2f}, "
            f"where 1.0 is exactly right")
    return out, note


def per_host_cascade_split(min_pages: int = 50) -> tuple[dict[str, dict], str]:
    """The same fit against the mass the Disconnect list does *not* name.

    `per_host_cascade_bytes` fits a page's tracker mass, so what it measures
    is the part of a host's cascade that lands on hosts the list names -- the
    part the crawl's quiet instrument can see. This runs the identical design
    against `tp_other_bytes`, the page's third-party mass the list does not
    name, with a free intercept for the baseline every page carries. The two
    coefficients together would say what share of a tracker's cascade the
    Disconnect-only delta can see.

    They do not, and the reason is worth keeping rather than deleting. The
    unlisted fit has no counterfactual in it: HTTP Archive never blocked
    anything, so a coefficient is what co-occurs with a host, and third-party
    mass co-occurs with *everything*. It credits www.google-analytics.com --
    a 10 KB beacon script -- with 255 KB of third-party bytes, and
    browser.sentry-cdn.com with 1.6 MB, which is not a cascade but a
    statement that heavy pages carry more of both. Summed over the fitted
    hosts it reads a listed share of 0.07 against the 0.40-0.63 the crawl
    measures two ways. Reported for that reason: the 50% export can identify
    what a tracker brings *within* the list, and cannot identify what it
    brings outside it.

    Returns (per-host listed/unlisted cascade bytes, a one-line description).
    """
    import numpy as np
    import scipy.sparse as sp

    con = _con()
    p = _ha50_presence(min_pages)
    k = len(p["hosts"])
    mass = con.sql(f"select page_domain, tp_other_bytes from "
                   f"'{HA50_MASS_CACHE}'").df()
    unlisted = (mass.set_index("page_domain").reindex(p["page_index"])
                .fillna(0.0).tp_other_bytes.to_numpy(float))
    unlisted = np.minimum(unlisted, np.quantile(unlisted, 0.999)) / 1024.0

    listed_design = sp.hstack([p["x"],
                               sp.csr_matrix(p["other"].reshape(-1, 1))]).tocsr()
    b_listed = _nnls_fit(listed_design, p["y_listed"])
    b_unlisted = _nnls_fit(
        sp.hstack([listed_design,
                   sp.csr_matrix(np.ones((len(unlisted), 1)))]).tocsr(),
        unlisted)

    out = {}
    for h, i in p["idx"].items():
        a = p["agg"].loc[h]
        listed = max(b_listed[i] - a.own, 0.0) * 1024.0
        out[h] = dict(listed=listed, unlisted=b_unlisted[i] * 1024.0,
                      listed_per_req=listed / max(a.reqs, 1.0),
                      unlisted_per_req=b_unlisted[i] * 1024.0 / max(a.reqs, 1.0))
    total_l = sum(d["listed"] for d in out.values())
    total_u = sum(d["unlisted"] for d in out.values())
    note = (f"{k} hosts; summed over them the fit puts "
            f"{total_l / (total_l + total_u):.2f} of the cascade on listed "
            f"hosts, against the 0.40-0.63 the crawl measures -- the page "
            f"intercept alone is {b_unlisted[k + 1] * 1024 / 1e3:.0f} KB, "
            f"which is the confounding talking")
    return out, note


# --------------------------------------------------------------------------- #
# The crawl: the counterfactual, on 271 pages
# --------------------------------------------------------------------------- #
def crawl_panel() -> list[dict]:
    """One record per paired page: what was blocked, and what the page shed."""
    blocked: dict[int, list] = collections.defaultdict(list)
    for r in top500.load_all_tracking_blocks():
        blocked[r.page_idx].append(r)

    panel = []
    for page in top500.load_pages():
        requests = blocked.get(page.idx, [])
        sizes, cpu_ms = top500.predict(requests) if requests else ([], [])
        # Zero-byte estimates are excluded because the estimator excludes
        # them: no body means nothing ran, so `followup_bytes_for` charges
        # them no subtree. Counting them here would measure a per-request
        # figure against a denominator the shipped code does not use.
        casc_sizes = [s for r, s in zip(requests, sizes)
                      if _ctx(r.resource_type) in CASCADING and s > 0]
        casc_urls = [(r.url, _ctx(r.resource_type) in CASCADING and s > 0)
                     for r, s in zip(requests, sizes)]
        panel.append(dict(
            page=page,
            direct=sum(sizes),
            direct_cpu_s=sum(cpu_ms) / 1000.0,
            casc=sum(casc_sizes),
            casc_sizes=casc_sizes,
            casc_urls=casc_urls,
            casc_types=[r.resource_type for r in requests],
            hosts=[(urlsplit(r.url).netloc.split(":")[0].lower(), s,
                    _ctx(r.resource_type) in CASCADING)
                   for r, s in zip(requests, sizes)],
        ))
    return panel


def factor_from_delta(panel: list[dict], corrected: bool = True) -> float:
    """The factor implied by the page-level byte delta over `panel`."""
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    measured = top500.measure([p["page"] for p in touched])
    saved = measured.bytes_saved if corrected else measured.bytes_saved_raw
    casc = sum(p["casc"] for p in touched)
    return (saved - sum(p["direct"] for p in touched)) / casc if casc else math.nan


def listed_factor_from_delta(panel: list[dict], corrected: bool = True) -> float:
    """The factor implied by the *Disconnect-only* page delta over `panel`.

    The same arithmetic as `factor_from_delta` against a narrower measurement:
    `bytes_saved_tracking` counts only the requests the Disconnect list names,
    so what comes out is the part of the cascade that lands back on listed
    hosts rather than all of it. Two things make it the sharpest cascade
    number in the repo. Its denominator is quiet -- across the crawl's ten
    passes it moves by 16.2% where the whole-page delta moves by 74%, so
    pooled they carry 5.1% and 23.4% -- and `request_set_difference` measures
    the same quantity a second way, from the HAR request sets rather than
    from the byte totals, and agrees.

    It *is* the shipped figure, as of the ten-pass crawl, and that is a
    change: the constant used to come from the whole-page delta, which on ten
    passes returns a value below this one despite counting a superset of the
    same requests. What it still is not is the whole cascade -- the rest of a
    blocked tracker's subtree lands on ad creatives, iframes and CDNs no list
    names, and no instrument here can see those. So the shipped figure is a
    lower bound. See `FOLLOWUP_BYTES_PER_REQUEST`.
    """
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    measured = top500.measure([p["page"] for p in touched])
    saved = (measured.bytes_saved_tracking if corrected
             else measured.bytes_saved_tracking_raw)
    casc = sum(p["casc"] for p in touched)
    return (saved - sum(p["direct"] for p in touched)) / casc if casc else math.nan


def _touched(panel: list[dict], stable: bool = False) -> list[dict]:
    """The pages ETP touched, optionally only those whose loads repeat.

    `stable` drops the pages `top500.page_delta_instability` scores above
    `top500.MAX_PAGE_INSTABILITY` -- six of 271, whose two arms differ load to
    load by about as much as the page weighs. It matters to exactly one
    measurement here and it matters to it a great deal; see
    `bytes_per_request_from_delta`.
    """
    out = [p for p in panel if p["page"].n_blocked_tracking > 0]
    if stable:
        unstable = top500.unstable_pages()
        out = [p for p in out if p["page"].idx not in unstable]
    return out


def bytes_per_request_from_delta(panel: list[dict], corrected: bool = True,
                                 listed: bool = False,
                                 stable: bool = False) -> float:
    """The same measurement in the units the array ships in: bytes a request.

    `factor_from_delta` divides the page-level cascade by the *bytes* of the
    blocked requests that can cascade; this divides by how many there were.
    Which of the two is the right denominator is the question `form_report`
    settles, and the answer is this one.

    `stable` is what makes the whole-page instrument usable, and it is the
    reason the shipped constant's justification changed without the constant
    moving. Over all 271 touched pages this reads 26.1 KB drift-corrected,
    95% [-26, 70] -- below the listed-only 47.0 KB that counts a subset of
    the same bytes, which is impossible and is what forced the shipped figure
    to be labelled a lower bound off one instrument with the other declared
    broken. Over the 265 whose loads repeat it reads 47.6 KB, 95% [12, 85].
    The listed-only figure barely moves under the same filter, 47.0 to 46.1,
    which is the check that this removes churn rather than cascade.

    So the two instruments now agree, at 47.6 and 46.1 against a shipped 47,
    and the impossibility is gone. What that does *not* do is measure the
    unlisted half: whole-page minus listed-only is 1.5 KB with an interval
    tens of KB wide, so the honest reading is that the loud instrument has
    stopped contradicting the quiet one, not that it has found the missing
    bytes. `Bounds.bytes_page_ratio_max_stable` in `tests/top500.py` is the
    bound that rests on the same filter.
    """
    touched = _touched(panel, stable)
    measured = top500.measure([p["page"] for p in touched])
    if listed:
        saved = (measured.bytes_saved_tracking if corrected
                 else measured.bytes_saved_tracking_raw)
    else:
        saved = measured.bytes_saved if corrected else measured.bytes_saved_raw
    n = sum(len(p["casc_sizes"]) for p in touched)
    return (saved - sum(p["direct"] for p in touched)) / n if n else math.nan


def bootstrap_per_request_ci(panel: list[dict], n: int = 4000,
                             seed: int = 7, listed: bool = False,
                             corrected: bool = True,
                             stable: bool = False) -> tuple[float, float]:
    """95% interval for `bytes_per_request_from_delta`, resampling pages."""
    touched = _touched(panel, stable)
    quiet = [p["page"] for p in panel if p["page"].n_blocked_tracking == 0]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sample = [touched[rng.randrange(len(touched))] for _ in touched]
        q = [quiet[rng.randrange(len(quiet))] for _ in quiet]
        m = top500.measure([p["page"] for p in sample], untouched=q)
        k = sum(len(p["casc_sizes"]) for p in sample)
        if not k:
            continue
        if listed:
            saved = (m.bytes_saved_tracking if corrected
                     else m.bytes_saved_tracking_raw)
        else:
            saved = m.bytes_saved if corrected else m.bytes_saved_raw
        out.append((saved - sum(p["direct"] for p in sample)) / k)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def bootstrap_factor_ci(panel: list[dict], n: int = 4000,
                        seed: int = 7) -> tuple[float, float]:
    """95% interval for `factor_from_delta`, resampling pages."""
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    quiet = [p["page"] for p in panel if p["page"].n_blocked_tracking == 0]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sample = [touched[rng.randrange(len(touched))] for _ in touched]
        q = [quiet[rng.randrange(len(quiet))] for _ in quiet]
        m = top500.measure([p["page"] for p in sample], untouched=q)
        casc = sum(p["casc"] for p in sample)
        if casc:
            out.append((m.bytes_saved - sum(p["direct"] for p in sample)) / casc)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def request_set_difference() -> dict[int, tuple[int, int]]:
    """Per page, (bytes, tracker bytes) of control-arm requests the other arm lacks.

    The blocking arm still *records* a blocked request -- it is in its HAR with
    no transfer -- so a directly blocked URL has a counterpart here and is not
    counted. What is left is what the blocking arm never asked for at all: the
    cascade, plus whatever the two loads differed by on their own, minus any
    descendant both arms happened to request. A lower bound.

    Summed over every pass of the crawl, which is what makes these totals
    comparable to the ones in `top500.load_pages`: those are pooled the same
    way, so a per-page figure here is ten passes' worth and divides into a
    ten-pass denominator. `top500.har_pairs` is what knows the layout.
    """
    import disconnect

    seen: dict[str, bool] = {}

    def is_tracker(url: str) -> bool:
        if url not in seen:
            try:
                seen[url] = bool(disconnect.is_tracker(url))
            except Exception:
                seen[url] = False
        return seen[url]

    out: dict[int, tuple[int, int]] = {}
    for page in top500.load_pages():
        extra = extra_tracker = 0
        pairs = top500.har_pairs(page.idx)
        if not pairs:
            continue
        for normal, private in pairs:
            other = collections.Counter(u for u, _ in load_har_sizes(private))
            for url, size in load_har_sizes(normal):
                if other[url] > 0:
                    other[url] -= 1
                    continue
                extra += size
                if is_tracker(url):
                    extra_tracker += size
        out[page.idx] = (extra, extra_tracker)
    return out


# --------------------------------------------------------------------------- #
# The form: per request or per byte
# --------------------------------------------------------------------------- #
def _instrument(panel: list[dict], listed: bool):
    """(observed cascade per page, the pages it covers), drift taken out."""
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    quiet = [p["page"] for p in panel if p["page"].n_blocked_tracking == 0]
    sel = ((lambda pg: pg.bytes_saved_tracking) if listed
           else (lambda pg: pg.bytes_saved))
    drift = st.fmean([sel(pg) for pg in quiet])
    return {p["page"].idx: sel(p["page"]) - drift - p["direct"]
            for p in touched}, touched


#: Consent-management platforms. Disconnect names them, so they are blocked
#: and charged a cascade like any other listed script, and `cmp_cascade` is
#: the test of whether that is right. See its docstring.
CMP_HOSTS = (
    "cookielaw.org", "osano.com", "privacy-mgmt.com", "cookiebot.com",
    "consentmanager.net", "onetrust.com", "trustarc.com", "quantcast.mgr",
    "usercentrics.eu", "cookieyes.com", "iubenda.com", "didomi.io",
)


def _is_cmp(host: str) -> bool:
    h = host.lower()
    return any(h == d or h.endswith("." + d) for d in CMP_HOSTS)


def cmp_cascade(panel: list[dict], n: int = 3000, seed: int = 0) -> None:
    """Consent managers as their own group: are they a subtree or a gate?

    `src/live_ablation.py` raised this. Ablating a CMP from a live page does
    not prune a subtree, it changes the page's consent state, and every ad
    decision downstream moves with it -- in whichever direction that site
    defaults. Six single-host ablations read +3.6 MB on braze.com, +199 KB on
    disneyplus.com, and -634 KB on ted.com, -104 KB on fwmrm.net, -56 KB on
    scribd.com: reproducible to a few kB each, and not the same sign. A
    conditional mean fitted over a population that mixes the two is fitting
    the crawl's accident of which sites default which way.

    That was six observations on six pages, which settles nothing. This
    re-asks it on the whole crawl with the same machinery
    `per_category_cascade` uses -- one row per page, a column counting the
    page's cascading CMP blocks and a column counting the rest, non-negative
    least squares against the quiet Disconnect-only instrument -- so that the
    answer comes from 271 pages rather than from six.

    Read the interval, not the point. If the CMP column's interval contains
    the other column's, this crawl cannot separate them and the shipped single
    figure stands, exactly as it does for tracker category.
    """
    import numpy as np
    from scipy.optimize import nnls

    obs, touched = _instrument(panel, True)
    ids = [p["page"].idx for p in touched]
    y = np.array([obs[i] for i in ids], float)
    x = np.zeros((len(ids), 2))
    for j, pg in enumerate(touched):
        for host, _size, casc in pg["hosts"]:
            if casc:
                x[j, 0 if _is_cmp(host) else 1] += 1

    n_cmp, n_rest = int(x[:, 0].sum()), int(x[:, 1].sum())
    pages_with = int((x[:, 0] > 0).sum())
    print(f"  {n_cmp:,} cascading blocks on consent managers over "
          f"{pages_with} pages; {n_rest:,} on everything else")
    if n_cmp < 25:
        print("  too few to fit; nothing to say")
        return

    b = nnls(x, y)[0]
    rng = random.Random(seed)
    draws = [[], []]
    for _ in range(n):
        pick = [rng.randrange(len(ids)) for _ in ids]
        bb = nnls(x[pick], y[pick])[0]
        draws[0].append(bb[0])
        draws[1].append(bb[1])
    for k, lbl in enumerate(("consent managers", "every other host")):
        d = sorted(draws[k])
        print(f"  {lbl:20s} {b[k]/1e3:8.1f} KB per request "
              f"95% [{d[int(.025*n)]/1e3:6.1f}, {d[int(.975*n)]/1e3:6.1f}]")
    pooled = nnls(x.sum(axis=1, keepdims=True), y)[0][0]
    print(f"  {'pooled (shipped form)':20s} {pooled/1e3:8.1f} KB per request")


def joint_fit(panel: list[dict], listed: bool = True,
              within_category: bool = False, n: int = 2000,
              seed: int = 0) -> dict:
    """Fit bytes-per-request and bytes-per-byte at once, and let them compete.

    Two regressors, one page per row: how many of the page's blocked requests
    could cascade, and how many bytes those requests were. Non-negative least
    squares, because a negative cascade is not a thing the model can mean.
    Whichever term the crawl actually needs keeps its coefficient and the other
    goes to zero -- which is the whole experiment, and it comes out the same
    way with the page's category demeaned out, so it is not the crawl's mix of
    page kinds talking.
    """
    import numpy as np
    from scipy.optimize import nnls

    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    y = np.array([obs[i] for i in ids], float)
    x = np.array([[len(p["casc_sizes"]), sum(p["casc_sizes"])] for p in touched],
                 float)
    groups = [p["page"].category for p in touched]

    def fit(yy, xx, gg):
        if gg is not None:
            yy, xx = yy.copy(), xx.copy()
            for c in set(gg):
                rows = [j for j, v in enumerate(gg) if v == c]
                yy[rows] -= yy[rows].mean()
                xx[rows] -= xx[rows].mean(axis=0)
        return nnls(xx, yy)[0]

    gg = groups if within_category else None
    beta = fit(y, x, gg)
    rng = random.Random(seed)
    draws = [[], []]
    for _ in range(n):
        pick = [rng.randrange(len(ids)) for _ in ids]
        b = fit(y[pick], x[pick], [groups[j] for j in pick] if gg else None)
        draws[0].append(b[0])
        draws[1].append(b[1])
    for d in draws:
        d.sort()
    lo, hi = int(0.025 * n), int(0.975 * n)
    return dict(per_request=beta[0], per_byte=beta[1],
                per_request_ci=(draws[0][lo], draws[0][hi]),
                per_byte_ci=(draws[1][lo], draws[1][hi]))


def by_request_size(panel: list[dict], edges=(10_000, 50_000),
                    listed: bool = True, n: int = 2000, seed: int = 0) -> list:
    """The implied per-byte factor, cut by the blocked request's own size.

    A proportional cascade says these three are the same number. They are not,
    and the direction is the one a per-request cascade predicts: the smaller
    the request, the more subtree per byte it appears to bring, because the
    subtree does not scale with it.
    """
    import numpy as np
    from scipy.optimize import nnls

    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    y = np.array([obs[i] for i in ids], float)
    x = np.zeros((len(ids), len(edges) + 1))
    for j, p in enumerate(touched):
        for size in p["casc_sizes"]:
            bucket = sum(1 for e in edges if size >= e)
            x[j, bucket] += size
    beta = nnls(x, y)[0]
    rng = random.Random(seed)
    draws = [[] for _ in range(x.shape[1])]
    for _ in range(n):
        pick = [rng.randrange(len(ids)) for _ in ids]
        b = nnls(x[pick], y[pick])[0]
        for k in range(x.shape[1]):
            draws[k].append(b[k])
    reversed_order = sum(1 for a, b_ in zip(draws[0], draws[-1]) if a < b_) / n
    out = []
    for k in range(x.shape[1]):
        d = sorted(draws[k])
        out.append((x[:, k].sum(), beta[k], d[int(0.025 * n)], d[int(0.975 * n)]))
    return out, reversed_order


def grade_form(panel: list[dict], listed: bool, n: int = 4000,
               seed: int = 0) -> dict:
    """Per request against per byte, with the crawl's cascade total held fixed.

    Neither model is allowed to win by being bigger: both are scaled to the
    same crawl-wide cascade, so what is being compared is where they put it.
    """
    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    total = SHIPPED_BYTES_PER_REQUEST * sum(len(p["casc_sizes"]) for p in touched)
    raw = {
        "per request": {p["page"].idx: float(len(p["casc_sizes"])) for p in touched},
        "per byte": {p["page"].idx: float(sum(p["casc_sizes"])) for p in touched},
    }
    pred = {}
    for name, v in raw.items():
        k = total / sum(v.values())
        pred[name] = {i: v[i] * k for i in ids}

    def errs(sample, cascade):
        page = sum(abs(cascade[i] - obs[i]) for i in sample)
        a, o = collections.Counter(), collections.Counter()
        for i in sample:
            a[cats[i]] += cascade[i]
            o[cats[i]] += obs[i]
        return page, sum(abs(a[c] - o[c]) for c in o)

    rng = random.Random(seed)
    win_page = win_cat = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        p1, c1 = errs(sample, pred["per request"])
        p0, c0 = errs(sample, pred["per byte"])
        win_page += p1 < p0
        win_cat += c1 < c0
    page_r, cat_r = errs(ids, pred["per request"])
    page_b, cat_b = errs(ids, pred["per byte"])
    return dict(page_per_request=page_r, page_per_byte=page_b,
                cat_per_request=cat_r, cat_per_byte=cat_b,
                win_page=win_page / n, win_cat=win_cat / n)


def saturation(panel: list[dict], listed: bool = True) -> list:
    """Bytes of cascade per blocked request, by how many were blocked there.

    NOT SHIPPED, and the one loose end the shipped form leaves. Two blocked
    loaders on a page do not prune two disjoint subtrees -- they share an ad
    stack -- so a strictly linear count should over-charge a heavily blocked
    page.

    Bucketed on the *per-pass* count, which is what the buckets were always
    meant to mean. The edges predate the ten-pass crawl and were read
    straight off `len(casc_sizes)`, which pooling multiplied by ten: every
    page moved up several buckets and 190 of the 200 landed in "8+", so the
    table this printed was one bucket and four rounding errors. Dividing by
    `top500.n_passes()` restores the grain the edges were chosen for and is a
    no-op on a crawl that was not repeated.

    Re-measured that way the hint is much stronger than the single pass could
    show -- 108 KB a request on the 55 pages with one cascading block against
    24 to 69 on the rest, where one pass had 91.7 KB on a single page -- and
    it still does not ship, because `grade_saturation` puts it head to head
    with the flat form at a fixed total and it loses on the quiet instrument
    at every exponent. See there.
    """
    obs, touched = _instrument(panel, listed)
    passes = top500.n_passes()
    out = []
    for label, lo, hi in (("1", 0.5, 1.5), ("2-3", 1.5, 3.5),
                          ("4-7", 3.5, 7.5), ("8-15", 7.5, 15.5),
                          ("16+", 15.5, float("inf"))):
        group = [p for p in touched
                 if lo <= len(p["casc_sizes"]) / passes < hi]
        k = sum(len(p["casc_sizes"]) for p in group)
        if k:
            out.append((label, len(group), round(k / passes),
                        sum(obs[p["page"].idx] for p in group) / k))
    return out


def intercept_fit(panel: list[dict], listed: bool = True, n: int = 3000,
                  seed: int = 0) -> dict:
    """Split the cascade into a per-page term and a per-request term.

    Every figure in this file charges the cascade per cascading blocked
    request and nothing per page. That is an assumption, not a measurement,
    and `grade_saturation` already concedes the crawl cannot tell a page-level
    intercept from a concave count -- any per-page offset divided by k falls
    in k exactly as saturation does.

    Single-host ablation can tell them apart, because it prices one host's
    subtree with the rest of the page running, and it disagrees sharply: over
    two 125-page runs it reads the listed cascade at 14.6 and 14.9 KB a
    request where this crawl's delta reads 47.0. Both are stable; they cannot
    both be per-request subtrees. If much of the 47.0 is really a page-level
    term -- bytes that go when a page's tracking is blocked at all, not bytes
    any particular request would have fetched -- then ablation, which can only
    see actual subtrees, reads low exactly the way it does.

    So fit both at once: one row per page, a column of ones and a column
    counting the page's cascading blocks, non-negative least squares. If the
    intercept takes most of the mass the shipped per-request form is
    mis-specified, whatever its total comes to.

    Also fitted against the *raw* delta, because the corrected one has already
    had the mean of the untouched pages subtracted, and the untouched pages
    are systematically the simpler ones -- so a spurious intercept could be a
    mis-estimated drift rather than a real per-page cost.

    IT DOES NOT SETTLE IT, and the way it fails is the interesting part. On
    the quiet instrument the intercept takes 361.6 KB a page and pulls the
    per-request term from 47.0 down to 33.2 KB -- a third of the way to what
    ablation reads, in the predicted direction, with 29% of the mass on the
    page term. But its 95% interval is [0, 800] KB and so contains zero; the
    two-term form takes 77% of paired bootstraps on per-page error, under the
    83% this file treats as noise, and loses outright on category totals at
    17%; and on the loud instrument the intercept goes to the boundary at
    exactly zero. Dropping the drift correction moves the intercept 361.6 ->
    274.4 and leaves the per-request term at 33.2, so it is not an artifact of
    correcting against the simpler untouched pages -- but that is the only
    thing here that is established.

    So the shipped per-request form stands, and the disagreement with ablation
    stands with it. Three explanations for it have now been tested and none is
    sufficient: interaction between blocked hosts (the joint arm reads 0.48 of
    the sum of the parts, so removal *over*counts rather than under), the
    denominator (counting only cascading aborts moves it 8%), and this. What
    would settle it is an instrument that prices a subtree and a joint
    counterfactual at once, which neither of the two here does.
    """
    import numpy as np
    from scipy.optimize import nnls

    out: dict = {}
    for corrected in (True, False):
        touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
        quiet = [p["page"] for p in panel
                 if p["page"].n_blocked_tracking == 0]
        sel = ((lambda pg: pg.bytes_saved_tracking) if listed
               else (lambda pg: pg.bytes_saved))
        drift = st.fmean([sel(pg) for pg in quiet]) if corrected else 0.0
        ids = [p["page"].idx for p in touched]
        y = np.array([sel(p["page"]) - drift - p["direct"] for p in touched],
                     float)
        x = np.array([[1.0, float(len(p["casc_sizes"]))] for p in touched])

        b = nnls(x, y)[0]
        rng = random.Random(seed)
        draws = [[], []]
        for _ in range(n):
            pick = [rng.randrange(len(ids)) for _ in ids]
            bb = nnls(x[pick], y[pick])[0]
            draws[0].append(bb[0])
            draws[1].append(bb[1])
        k_tot = x[:, 1].sum()
        out["corrected" if corrected else "raw"] = dict(
            intercept=b[0], per_request=b[1],
            intercept_ci=(sorted(draws[0])[int(.025 * n)],
                          sorted(draws[0])[int(.975 * n)]),
            per_request_ci=(sorted(draws[1])[int(.025 * n)],
                            sorted(draws[1])[int(.975 * n)]),
            page_mass=b[0] * len(ids), request_mass=b[1] * k_tot,
            n_pages=len(ids), n_requests=k_tot)
    return out


def grade_intercept(panel: list[dict], listed: bool, n: int = 4000,
                    seed: int = 0) -> dict:
    """The two-term form against the shipped one, at a fixed crawl total.

    Same harness as `grade_form` and `grade_saturation`: level both models to
    the same crawl-wide cascade so neither can win by being bigger, then
    compare per-page error and per-category totals over paired bootstraps.
    Without the levelling a model with an extra free parameter wins on total
    alone and says nothing about shape.
    """
    import numpy as np
    from scipy.optimize import nnls

    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    y = np.array([obs[i] for i in ids], float)
    x = np.array([[1.0, float(len(p["casc_sizes"]))] for p in touched])
    b = nnls(x, y)[0]

    flat = {i: float(len(p["casc_sizes"]))
            for i, p in zip(ids, touched)}
    cand = {i: b[0] + b[1] * len(p["casc_sizes"])
            for i, p in zip(ids, touched)}
    total = SHIPPED_BYTES_PER_REQUEST * sum(flat.values())
    for d in (flat, cand):
        k = total / sum(d.values())
        for i in d:
            d[i] *= k

    def errs(sample, cascade):
        page = sum(abs(cascade[i] - obs[i]) for i in sample)
        a, o = collections.Counter(), collections.Counter()
        for i in sample:
            a[cats[i]] += cascade[i]
            o[cats[i]] += obs[i]
        return page, sum(abs(a[c] - o[c]) for c in o)

    rng = random.Random(seed)
    win_page = win_cat = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        p1, c1 = errs(sample, cand)
        p0, c0 = errs(sample, flat)
        win_page += p1 < p0
        win_cat += c1 < c0
    return dict(win_page=win_page / n, win_cat=win_cat / n,
                page_flat=errs(ids, flat)[0], page_cand=errs(ids, cand)[0])


def grade_saturation(panel: list[dict], listed: bool, gamma: float,
                     n: int = 4000, seed: int = 0) -> dict:
    """A concave count against a linear one, with the crawl's total held fixed.

    The test `saturation` above cannot do for itself. A bucket table shows the
    per-request figure falling as the count rises, but that is also what a
    page-level intercept looks like -- any per-page offset divided by k falls
    in k -- so the shape has to be graded where it would actually be applied:
    charge `k ** gamma` instead of `k`, level both models to the same
    crawl-wide cascade so neither can win by being bigger, and compare.

    It loses. On the quiet instrument the flat form is better on both metrics
    at every exponent tried, and the best the concave form manages is 57% of
    paired bootstraps on per-page error at gamma 0.95, which is to say by
    being almost exactly the flat form. The loud instrument likes it a little
    more (74% at 0.95) and the loud instrument is the one with four times the
    standard error. Nothing here clears the 83% this file treats as noise.

    So the bucket table is a page-level intercept the crawl cannot separate
    from a concave count, which is the honest form of "left open".

    Exponents above 1 lose as well, and they were worth trying because a
    different instrument pointed at them. Single-host ablation
    (`src/live_ablation.py`) prices each host's subtree on its own, so a
    page-level intercept cannot divide into it the way it does into the bucket
    table here -- and measured that way the per-request cascade *rises* with
    how many requests the page has blocked, 10.2 to 76.5 to 87.4 KB across
    count buckets, which is the opposite shape. Graded here it does not hold:
    on the quiet instrument gamma 1.05 takes 45% of paired bootstraps on
    per-page error and 1.10 takes 44%, both worse than flat, with category
    totals at 68% and 67% and so under the 83% this file treats as noise; on
    the loud instrument 1.10 takes 11%. Flat wins against curvature in both
    directions, which is a stronger statement than the concave test alone
    made.
    """
    obs, touched = _instrument(panel, listed)
    passes = top500.n_passes()
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    flat = {p["page"].idx: float(len(p["casc_sizes"])) for p in touched}
    cand = {p["page"].idx: passes * (len(p["casc_sizes"]) / passes) ** gamma
            for p in touched}
    total = SHIPPED_BYTES_PER_REQUEST * sum(flat.values())
    for d in (flat, cand):
        k = total / sum(d.values())
        for i in d:
            d[i] *= k

    def errs(sample, cascade):
        page = sum(abs(cascade[i] - obs[i]) for i in sample)
        a, o = collections.Counter(), collections.Counter()
        for i in sample:
            a[cats[i]] += cascade[i]
            o[cats[i]] += obs[i]
        return page, sum(abs(a[c] - o[c]) for c in o)

    rng = random.Random(seed)
    win_page = win_cat = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        p1, c1 = errs(sample, cand)
        p0, c0 = errs(sample, flat)
        win_page += p1 < p0
        win_cat += c1 < c0
    page_f, cat_f = errs(ids, flat)
    page_c, cat_c = errs(ids, cand)
    return dict(page_flat=page_f, page_concave=page_c, cat_flat=cat_f,
                cat_concave=cat_c, win_page=win_page / n, win_cat=win_cat / n)


def per_pass_cascade_hosts() -> dict[int, list[list[str]]] | None:
    """Per page, per pass, the hosts of the blocked requests that can cascade.

    The one measurement in this file that needs the crawl's passes kept apart
    rather than pooled, and it is easy to get wrong in a way that flatters the
    answer. A host blocked once per pass on a page appears ten times in the
    pooled table; deduplicating there would collapse the ten passes as well as
    the within-page repeats and report a 92% reduction where the truth is 26%.
    So this reads `paired/rep*/blocked_observed_bytes.csv` and keeps the
    passes separate, and `host_dedup` sums over them.

    None when the per-pass tables are absent, which is a single-pass crawl.
    """
    reps = top500.reps()
    paths = [top500.CRAWL_DIR / "paired" / r / "blocked_observed_bytes.csv"
             for r in reps]
    if len(paths) < 2 or not all(q.exists() for q in paths):
        return None
    import llm_classifier as lc
    out: dict[int, list[list[str]]] = collections.defaultdict(list)
    for path in paths:
        per_page: dict[int, list[str]] = collections.defaultdict(list)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["protection"] != "tracking":
                    continue
                if _ctx(row["resource_type"]) not in CASCADING:
                    continue
                size, _ = lc.estimate_resources(
                    row["blocked_url"], context_for(row["resource_type"]),
                    lc.RequestInitiator.UNKNOWN, "", False)
                if size <= 0:
                    continue
                per_page[int(row["page_idx"])].append(
                    urlsplit(row["blocked_url"]).netloc.split(":")[0].lower())
        for idx, hosts in per_page.items():
            out[idx].append(hosts)
    return dict(out)


def host_dedup(panel: list[dict], by_pass: dict[int, list[list[str]]],
               listed: bool, repeat_weight: float = 0.0,
               n: int = 4000, seed: int = 0, permute: bool = False) -> dict:
    """Charging the cascade per distinct tracker host against per request.

    SHIPPED, and the only refinement in this file that is. See
    `FOLLOWUP_BYTES_PER_HOST` in `src/lib.rs`.

    Two blocked requests to `doubleclick.net` on one page load prune one ad
    stack between them, not two. `saturation` asks the same question through
    the count of blocked requests and cannot answer it; this asks it through
    which hosts they went to, and the crawl answers clearly. Graded the way
    everything else here is -- levelled to a fixed crawl total, paired
    bootstrap over pages -- it wins 99% of resamples on per-page error and 86%
    on category totals against the quiet instrument, which no other refinement
    tried against this crawl manages on either metric, let alone both.

    `permute` is the control that makes the result mean something. A model
    charging fewer units on heavily blocked pages could be a saturation term
    wearing a disguise, so this keeps each page's per-pass count of cascading
    blocks and draws their hosts at random from the crawl-wide pool: the
    counts collapse by the same amount for no reason at all. Over 100 shuffles
    that wins about 48% of resamples on average, none of them reaches the real
    99%, and the shuffled model's per-page error lands on the flat model's to
    within 0.2 MB. The effect is which hosts, not how many requests.
    """
    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    rng = random.Random(seed)
    pool = [h for passes in by_pass.values() for hs in passes for h in hs]

    def units(idx: int, weight: float) -> float:
        total = 0.0
        for hosts in by_pass.get(idx, []):
            if permute:
                hosts = [pool[rng.randrange(len(pool))] for _ in hosts]
            for k in collections.Counter(hosts).values():
                total += 1 + weight * (k - 1)
        return total

    flat = {i: units(i, 1.0) for i in ids}
    cand = {i: units(i, repeat_weight) for i in ids}
    if not sum(flat.values()) or not sum(cand.values()):
        return {}
    total = SHIPPED_BYTES_PER_REQUEST * sum(flat.values())
    for d in (flat, cand):
        k = total / sum(d.values())
        for i in d:
            d[i] *= k

    def errs(sample, cascade):
        page = sum(abs(cascade[i] - obs[i]) for i in sample)
        a, o = collections.Counter(), collections.Counter()
        for i in sample:
            a[cats[i]] += cascade[i]
            o[cats[i]] += obs[i]
        return page, sum(abs(a[c] - o[c]) for c in o)

    boot = random.Random(seed + 1)
    win_page = win_cat = 0
    for _ in range(n):
        sample = [ids[boot.randrange(len(ids))] for _ in ids]
        p1, c1 = errs(sample, cand)
        p0, c0 = errs(sample, flat)
        win_page += p1 < p0
        win_cat += c1 < c0
    page_f, cat_f = errs(ids, flat)
    page_c, cat_c = errs(ids, cand)
    n_req = sum(units(i, 1.0) for i in ids)
    n_host = sum(units(i, 0.0) for i in ids)
    return dict(page_flat=page_f, page_host=page_c, cat_flat=cat_f,
                cat_host=cat_c, win_page=win_page / n, win_cat=win_cat / n,
                n_requests=n_req, n_hosts=n_host,
                per_request=sum(obs[i] for i in ids) / n_req,
                per_host=sum(obs[i] for i in ids) / n_host)


#: Second-level suffixes common enough that stripping to two labels would
#: merge unrelated registrants. Enough for `_registrable`, which only has to
#: be a fair alternative grain to grade the shipped one against, not a public
#: suffix list.
_SECOND_LEVEL = frozenset(
    ("co", "com", "org", "net", "ac", "gov", "edu", "or", "ne"))


def _registrable(host: str) -> str:
    """`host` collapsed to something close to its registrable domain."""
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def on_host_baseline(panel: list[dict], by_pass: dict[int, list[list[str]]],
                     n: int = 4000, seed: int = 0) -> None:
    """Re-grade the rejected shapes against the per-host model, not the flat one.

    Everything in the two NOT SHIPPED sections below was graded against
    charging one flat figure per cascading request, which is no longer what
    the crate does. A shape rejected on that baseline could in principle be
    picking up something the per-host rule now captures -- or could still have
    room beside it -- and only re-grading says which.

    It says the first. Saturation in the count collapses from 19-57% of
    bootstraps to 11-32% once the hosts are deduplicated, which is the honest
    reading of what `FOLLOWUP_BYTES_PER_HOST` is: saturation measured through
    the right variable. Size modulation improves, 24% to 71%, and still does
    not clear 83%.

    Also graded here, because it is the obvious alternative to the shipped
    grain: collapsing hosts to their registrable domain. It loses at 28%.
    """
    obs, touched = _instrument(panel, listed=True)
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    passes = top500.n_passes()

    def per_page(keyed, weight_of):
        out = collections.Counter()
        for idx in ids:
            for hosts in by_pass.get(idx, []):
                seen = set()
                for h in hosts:
                    k = keyed(h)
                    if k not in seen:
                        seen.add(k)
                out[idx] += weight_of(len(seen))
        return out

    base = per_page(lambda h: h, lambda k: float(k))

    def grade(cand, label):
        a = {i: float(base.get(i, 0.0)) for i in ids}
        b = {i: float(cand.get(i, 0.0)) for i in ids}
        total = sum(obs[i] for i in ids)
        for d in (a, b):
            t = sum(d.values())
            if not t:
                return
            for i in d:
                d[i] *= total / t

        def errs(sample, c):
            page = sum(abs(c[i] - obs[i]) for i in sample)
            x, o = collections.Counter(), collections.Counter()
            for i in sample:
                x[cats[i]] += c[i]
                o[cats[i]] += obs[i]
            return page, sum(abs(x[y] - o[y]) for y in o)

        rng = random.Random(seed)
        win_page = win_cat = 0
        for _ in range(n):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            p1, c1 = errs(sample, b)
            p0, c0 = errs(sample, a)
            win_page += p1 < p0
            win_cat += c1 < c0
        pa, ca = errs(ids, a)
        pb, cb = errs(ids, b)
        print(f"    {label:28s} page {pa/1e6:6.1f} -> {pb/1e6:6.1f} MB "
              f"{win_page/n:4.0%}   category {ca/1e6:5.1f} -> {cb/1e6:5.1f} MB "
              f"{win_cat/n:4.0%}")

    grade(per_page(_registrable, float), "per registrable domain")
    for gamma in (0.75, 0.85, 0.95):
        grade(per_page(lambda h: h, lambda k: passes * (k / passes) ** gamma
                       if k else 0.0), f"distinct hosts ** {gamma:.2f}")


def grade_by_size(panel: list[dict], listed: bool, delta: float,
                  n: int = 4000, seed: int = 0,
                  s_ref: float = 31_800.0) -> dict:
    """A size-modulated cascade against a flat one, with the total held fixed.

    The other half of the form question, and the one the shipped comment does
    not ask. `grade_form` compares per request against per byte -- exponents
    0 and 1 on the blocked request's own size -- and per request wins. That
    leaves everything between them untested, and the crawl does have a signal
    there: cut the cascade by the blocked request's size and the quiet
    instrument reads 28.0, 24.4, 54.6 and 51.4 KB a request over <5, 5-20,
    20-60 and 60+ KB, so a large blocked script cascades about twice what a
    small one does. Twice, not twenty times, which is why proportional loses.

    Charging `(size / s_ref) ** delta` tests that directly. It loses too: on
    the quiet instrument the flat form wins on per-page error at every delta
    from 0.10 up, and the category metric prefers a small delta at 67%, under
    the bar. The loud instrument likes it at 99%, and disagrees with the quiet
    one about the direction of everything in this file.

    `s_ref` is HTTP Archive's mean tracker script, so delta 0 and the shipped
    constant agree exactly and the comparison is shape against shape.
    """
    obs, touched = _instrument(panel, listed)
    ids = [p["page"].idx for p in touched]
    cats = {p["page"].idx: p["page"].category for p in touched}
    flat = {p["page"].idx: float(len(p["casc_sizes"])) for p in touched}
    cand = {p["page"].idx: sum((s / s_ref) ** delta for s in p["casc_sizes"])
            for p in touched}
    total = SHIPPED_BYTES_PER_REQUEST * sum(flat.values())
    for d in (flat, cand):
        k = total / sum(d.values())
        for i in d:
            d[i] *= k

    def errs(sample, cascade):
        page = sum(abs(cascade[i] - obs[i]) for i in sample)
        a, o = collections.Counter(), collections.Counter()
        for i in sample:
            a[cats[i]] += cascade[i]
            o[cats[i]] += obs[i]
        return page, sum(abs(a[c] - o[c]) for c in o)

    rng = random.Random(seed)
    win_page = win_cat = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        p1, c1 = errs(sample, cand)
        p0, c0 = errs(sample, flat)
        win_page += p1 < p0
        win_cat += c1 < c0
    page_f, cat_f = errs(ids, flat)
    page_c, cat_c = errs(ids, cand)
    return dict(page_flat=page_f, page_sized=page_c, cat_flat=cat_f,
                cat_sized=cat_c, win_page=win_page / n, win_cat=win_cat / n)


# --------------------------------------------------------------------------- #
# Refinements that are measured and not shipped
# --------------------------------------------------------------------------- #
def per_category(panel: list[dict], n: int = 4000, seed: int = 7) -> None:
    """Is the cascade different for a news site than for a bank?

    Very likely yes, and this crawl cannot see it. Printed with intervals
    rather than as point estimates for exactly that reason.
    """
    quiet = [dict(page=p["page"], direct=0, casc=0, casc_sizes=[], hosts=[])
             for p in panel if p["page"].n_blocked_tracking == 0]
    print(f"  {'category':15s} {'pages':>5s} {'requests':>9s} "
          f"{'KB/request':>11s} {'95% interval':>18s}")
    for category in top500.CATEGORIES + ("ALL",):
        group = [p for p in panel if p["page"].n_blocked_tracking > 0
                 and (category == "ALL" or p["page"].category == category)]
        if not group:
            continue
        point = bytes_per_request_from_delta(group + quiet)
        lo, hi = bootstrap_per_request_ci(group + quiet, n=n, seed=seed)
        print(f"  {category:15s} {len(group):5d} "
              f"{sum(len(p['casc_sizes']) for p in group):9d} "
              f"{point/1e3:11.1f} [{lo/1e3:7.0f},{hi/1e3:7.0f}]")


def crawl_roles(panel: list[dict]) -> dict[int, list[str]] | None:
    """The tracker *role* of each blocked request that can cascade.

    `crawl_categories` splits by the Disconnect list's category alone; this
    splits by `top500.tracker_role_for`, which crosses that with what the
    request was for. It is the axis `tests/top500.py` bounds the size
    estimate on, and the question here is whether it also carries a cascade:
    an ad loader plausibly prunes a larger subtree than an analytics script,
    and if the crawl could see that, the shipped constant should be several
    numbers rather than one.

    Only the cascading contexts reach this, so `kind` is `script` or `frame`
    throughout and the split is close to the one above. That is deliberate:
    it is the finer split, on exactly the population the constant applies to,
    and `per_category_cascade` prints what a finer split costs in resolution.
    """
    try:
        import top500 as t5
    except ImportError:
        return None
    out = {}
    for p in panel:
        if p["page"].n_blocked_tracking > 0:
            out[p["page"].idx] = [
                t5.tracker_role_for(u, rt)
                for (u, kept), rt in zip(p["casc_urls"], p["casc_types"])
                if kept]
    return out


def crawl_match_levels(panel: list[dict]) -> dict[int, list[str]] | None:
    """Which rung of the table answered for each block that can cascade.

    The axis `FOLLOWUP_RUNG_SCALE` is keyed on, and the only one of the four
    refinements tried against this crawl that ships anything at all.
    """
    try:
        import llm_classifier as lc
        import top500 as t5  # noqa: F401  (import guard only)
    except ImportError:
        return None
    out = {}
    for p in panel:
        if p["page"].n_blocked_tracking > 0:
            out[p["page"].idx] = [
                lc.classify_url(u, context_for(rt),
                                lc.RequestInitiator.UNKNOWN, "")
                for (u, kept), rt in zip(p["casc_urls"], p["casc_types"])
                if kept]
    return out


def rung_cascade_shape(panel: list[dict], by_rung: dict[int, list[str]],
                       extra: dict[int, tuple[int, int]] | None = None,
                       min_blocks: int = 100, n: int = 4000,
                       seed: int = 11) -> None:
    """Re-derive `FOLLOWUP_RUNG_SCALE`, dissent included.

    The shipped factors are the one refinement to the cascade that survived,
    and they survived weakly, so this prints the whole derivation rather than
    the answer: the fit on each instrument, how often the ordering holds
    under resampling, how much of the spread survives its own noise, and the
    levelled factors that come out. A reader who thinks the evidence is too
    thin should be able to see that from the output, not have to take it on
    trust.

    Three instruments. Two are quiet and agree; the third is the whole-page
    delta, which `top500` documents as too noisy at this grain and which
    orders `ext_query` the other way with equal confidence. It is printed
    for exactly that reason.
    """
    import numpy as np
    from scipy.optimize import nnls

    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    quiet = [p["page"] for p in panel if p["page"].n_blocked_tracking == 0]
    counts = collections.Counter(c for p in touched
                                 for c in by_rung[p["page"].idx])
    cols = [c for c, k in counts.most_common() if k >= min_blocks]
    x = np.zeros((len(touched), len(cols) + 1))
    for i, p in enumerate(touched):
        for c in by_rung[p["page"].idx]:
            x[i, cols.index(c) if c in cols else len(cols)] += 1

    def target(kind):
        if kind == "request set":
            if extra is None:
                return None
            drift = st.fmean([extra.get(pg.idx, (0, 0))[1] for pg in quiet])
            return np.array([extra.get(p["page"].idx, (0, 0))[1] - drift
                             for p in touched], float)
        obs, _ = _instrument(panel, listed=(kind == "listed-only delta"))
        return np.array([obs[p["page"].idx] for p in touched], float)

    def fit(design, y):
        return nnls(design.T @ design, design.T @ y)[0]

    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(touched), len(touched)) for _ in range(n)]
    labels = ("request set", "listed-only delta", "whole-page delta")
    fits, draws = {}, {}
    print(f"  {'instrument':20s}" + "".join(f"{c:>16s}" for c in cols))
    for kind in labels:
        y = target(kind)
        if y is None:
            print(f"  {kind:20s}  (skipped: --skip-request-set)")
            continue
        fits[kind] = fit(x, y)
        draws[kind] = np.array([fit(x[i], y[i]) for i in picks])
        print(f"  {kind:20s}"
              + "".join(f"{fits[kind][j]/1e3:13.1f} KB" for j in range(len(cols))))

    quiet_kinds = [k for k in labels[:2] if k in draws]
    print("\n  how often the ordering holds, resampling pages:")
    for a, b in itertools.combinations(range(len(cols)), 2):
        line = "   ".join(
            f"{k.split()[0]} {100*np.mean(draws[k][:, a] > draws[k][:, b]):5.1f}%"
            for k in draws)
        print(f"    P({cols[a]} > {cols[b]}):  {line}")

    if not quiet_kinds:
        return
    bhat = np.mean([fits[k] for k in quiet_kinds], axis=0)
    se = np.mean([draws[k].std(axis=0) for k in quiet_kinds],
                 axis=0) / len(quiet_kinds) ** 0.5
    weights = np.array([counts[c] for c in cols]
                       + [sum(v for c, v in counts.items() if c not in cols)],
                       float)
    between = max(0.0, np.average((bhat - SHIPPED_BYTES_PER_REQUEST) ** 2,
                                  weights=weights)
                  - np.average(se ** 2, weights=weights))
    w = between / (between + se ** 2)
    shrunk = SHIPPED_BYTES_PER_REQUEST + w * (bhat - SHIPPED_BYTES_PER_REQUEST)
    level = (SHIPPED_BYTES_PER_REQUEST * weights.sum()) / (shrunk * weights).sum()
    print(f"\n  between-rung sd surviving noise {between ** 0.5 / 1e3:.1f} KB "
          f"against a mean standard error of {np.mean(se) / 1e3:.1f} KB,"
          f"\n  so the fit is shrunk toward the single figure and levelled:")
    print(f"    {'rung':16s} {'blocks':>7s} {'fitted':>9s} {'weight':>7s} "
          f"{'factor':>8s}")
    for j, c in enumerate(cols + ["other"]):
        print(f"    {c:16s} {int(weights[j]):7,d} {bhat[j]/1e3:8.1f}K "
              f"{w[j]:7.2f} {shrunk[j]*level/SHIPPED_BYTES_PER_REQUEST:8.2f}")
    print("  compare FOLLOWUP_RUNG_SCALE in src/lib.rs, which rounds these to "
          "two places.")


def crawl_categories(panel: list[dict]) -> dict[int, list[str]] | None:
    """The Disconnect category of each blocked request that can cascade.

    None when the `disconnect` extension is not importable, which is the one
    dependency the rest of this script does not need.
    """
    try:
        import disconnect
    except ImportError:
        return None
    seen: dict[str, str] = {}

    def category(url: str) -> str:
        if url not in seen:
            try:
                got = disconnect.tracker_category(url)
            except Exception:
                got = None
            seen[url] = str(got) if got else "unlisted"
        return seen[url]

    out = {}
    for p in panel:
        if p["page"].n_blocked_tracking > 0:
            out[p["page"].idx] = [category(u) for u, kept in p["casc_urls"]
                                  if kept]
    return out


def per_category_cascade(panel: list[dict], by_category: dict[int, list[str]],
                         ha50_shape: dict | None = None, min_requests: int = 25,
                         n: int = 3000, seed: int = 0) -> None:
    """A figure per tracker category: what a bigger crawl would need to settle.

    The most natural refinement left, and the one the Disconnect list hands
    over for free: an Advertising tracker's subtree is an ad stack, an
    Analytics tracker's is a beacon, a Social one's is a widget. The crawl puts
    them in that order and cannot separate them. Fitted against the quiet
    instrument every category's interval contains the single figure, and
    against the loud one the order reverses with no power at all -- which is
    what an instrument with no resolution looks like rather than a refutation.

    The arithmetic of why: 7,089 cascading blocks over 271 pages support one
    figure to about a third of itself. Splitting into three widens each
    interval by roughly the root of three while the categories differ by less
    than that, so the split cannot pay for itself on this crawl whatever the
    truth is. A crawl ten times the size would separate 39 KB from 22 KB.

    Borrowing the shape from HTTP Archive instead does not work either; see
    `category_shape_from_ha50`. This grades it anyway, because "we could not
    borrow it" is worth a number too.
    """
    import numpy as np
    from scipy.optimize import nnls

    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    ids = [p["page"].idx for p in touched]
    groups = [p["page"].category for p in touched]
    direct = {p["page"].idx: p["direct"] for p in touched}
    counts = collections.Counter(c for i in ids for c in by_category[i])
    cats = [c for c, k in counts.most_common() if k >= min_requests]
    cols = cats + ["other"]
    x = np.zeros((len(ids), len(cols)))
    for j, i in enumerate(ids):
        for c in by_category[i]:
            x[j, cols.index(c) if c in cats else len(cats)] += 1

    def fit(y, design, gg=None):
        if gg is not None:
            y, design = y.copy(), design.copy()
            for g in set(gg):
                rows = [k for k, v in enumerate(gg) if v == g]
                y[rows] -= y[rows].mean()
                design[rows] -= design[rows].mean(axis=0)
        return nnls(design, y)[0]

    print(f"  {'category':22s} {'blocks':>7s} " + "".join(
        f"{lbl:>26s}" for lbl in ("listed-only delta", "whole-page delta")))
    obs = {}
    for listed in (True, False):
        obs[listed], _ = _instrument(panel, listed)
    fits, pooled = {}, {}
    for listed in (True, False):
        y = np.array([obs[listed][i] for i in ids], float)
        fits[listed] = fit(y, x)
        pooled[listed] = fit(y, x.sum(axis=1, keepdims=True))[0]
        rng = random.Random(seed)
        draws = [[] for _ in cols]
        for _ in range(n):
            pick = [rng.randrange(len(ids)) for _ in ids]
            b = fit(y[pick], x[pick])
            for k in range(len(cols)):
                draws[k].append(b[k])
        fits[(listed, "ci")] = [
            (sorted(d)[int(0.025 * n)], sorted(d)[int(0.975 * n)]) for d in draws]
    for k, c in enumerate(cols):
        row = f"  {c:22s} {int(x[:, k].sum()):7d} "
        for listed in (True, False):
            lo, hi = fits[(listed, "ci")][k]
            row += (f"{fits[listed][k]/1e3:8.1f} KB [{lo/1e3:4.0f},{hi/1e3:4.0f}]")
        print(row)
    print(f"  {'one figure for all':22s} {int(x.sum()):7d} " + "".join(
        f"{pooled[listed]/1e3:8.1f} KB {'':11s}" for listed in (True, False)))

    if ha50_shape:
        total = SHIPPED_BYTES_PER_REQUEST * int(x.sum())
        mean = st.fmean([ha50_shape.get(c, 0.0) for i in ids
                         for c in by_category[i]]) or 1.0
        raw = {i: sum(ha50_shape.get(c, mean) for c in by_category[i])
               for i in ids}
        k = total / max(sum(raw.values()), 1)
        cand = {i: direct[i] + raw[i] * k for i in ids}
        flat = {i: direct[i] + SHIPPED_BYTES_PER_REQUEST * len(by_category[i])
                for i in ids}
        cats_of = dict(zip(ids, groups))
        for listed in (True, False):
            o = obs[listed]

            def errs(sample, pred):
                a, b = collections.Counter(), collections.Counter()
                for i in sample:
                    a[cats_of[i]] += pred[i]
                    b[cats_of[i]] += o[i]
                return (sum(abs(pred[i] - o[i]) for i in sample),
                        sum(abs(a[g] - b[g]) for g in b))

            rng = random.Random(seed)
            wp = wc = 0
            for _ in range(n):
                sample = [ids[rng.randrange(len(ids))] for _ in ids]
                p1, c1 = errs(sample, cand)
                p0, c0 = errs(sample, flat)
                wp += p1 < p0
                wc += c1 < c0
            print(f"  the 50% export's shape, levelled and graded against the "
                  f"{INSTRUMENT['tracking' if listed else 'page']}:"
                  f" wins {wp/n:.0%} on per-page error, {wc/n:.0%} on category "
                  f"totals")


def grade_shape(panel: list[dict], cascade_of, label: str,
                n: int = 4000, seed: int = 0, target: str = "page") -> None:
    """Does a per-site cascade allocate the same total better than a flat one?

    Both models are rescaled so that direct plus cascade sums to the measured
    total, because the question is where the cascade goes and not how big it
    is. Rescaling *both* rather than matching the candidate to the flat model
    matters on the quiet target below, where the flat model's own level sits
    1.5x above the measurement by construction: leaving that in would score
    two models against an offset they share and call the difference shape.

    Judged two ways: absolute error per page, and absolute error over the
    category aggregates, which is closer to the sum a dashboard claims.
    Paired bootstrap over pages -- one resample per draw, both models scored
    on it, since resampling them separately compares the spread of two error
    totals rather than the models.

    `target` picks the instrument, and the choice decides how much the answer
    is worth:

      page      the drift-corrected whole-page delta. What the cascade is
                calibrated against, and the loudest thing here: 1.37 MB of
                churn per page against a mean saving of 0.29 MB.

      tracking  the drift-corrected Disconnect-only delta. Blind to the part
                of the cascade that lands off the list, so it can only grade
                shape, which is what this function asks about -- and it is
                the quiet one, at 0.18 MB of churn per page. Where a per-site
                shape has a chance of being visible at all, it is here.
    """
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    quiet = [p for p in panel if p["page"].n_blocked_tracking == 0]
    if target == "tracking":
        drift = st.fmean([p["page"].bytes_saved_tracking for p in quiet])
        observed = {p["page"].idx: p["page"].bytes_saved_tracking - drift
                    for p in touched}
    else:
        drift = st.fmean([p["page"].bytes_saved for p in quiet])
        observed = {p["page"].idx: p["page"].bytes_saved - drift
                    for p in touched}
    direct = {p["page"].idx: p["direct"] for p in touched}
    obs = observed
    cats = {p["page"].idx: p["page"].category for p in touched}
    flat = {p["page"].idx: SHIPPED_BYTES_PER_REQUEST * len(p["casc_sizes"])
            for p in touched}
    cand = {p["page"].idx: sum(cascade_of(h, b, c) for h, b, c in p["hosts"])
            for p in touched}
    print(f"  {label}")
    if not sum(cand.values()):
        print("    no coverage, skipped")
        return

    # Level the two models against the measurement, not against each other.
    room = sum(obs.values()) - sum(direct.values())
    flat = {i: v * room / sum(flat.values()) for i, v in flat.items()}
    cand = {i: v * room / sum(cand.values()) for i, v in cand.items()}
    ids = list(obs)

    def metrics(sample):
        mae_f = sum(abs(direct[i] + flat[i] - obs[i]) for i in sample)
        mae_c = sum(abs(direct[i] + cand[i] - obs[i]) for i in sample)
        af, ac, ao = (collections.Counter() for _ in range(3))
        for i in sample:
            af[cats[i]] += direct[i] + flat[i]
            ac[cats[i]] += direct[i] + cand[i]
            ao[cats[i]] += obs[i]
        return (mae_f, mae_c,
                sum(abs(af[c] - ao[c]) for c in ao),
                sum(abs(ac[c] - ao[c]) for c in ao))

    mf, mc, cf, cc = metrics(ids)
    rng = random.Random(seed)
    win_mae = win_cat = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        a, b, c, d = metrics(sample)
        win_mae += b < a
        win_cat += d < c
    print(f"    per-page |error|   shipped {mf/1e6:6.1f} MB   reweighted "
          f"{mc/1e6:6.1f} MB   reweighted wins {win_mae/n:.0%}")
    print(f"    category |error|   shipped {cf/1e6:6.1f} MB   reweighted "
          f"{cc/1e6:6.1f} MB   reweighted wins {win_cat/n:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-request-set", action="store_true",
                    help="skip the HAR method, which reads 1,000 files")
    ap.add_argument("--skip-50pct", action="store_true",
                    help="skip the per-site fit from the 50%% export")
    ap.add_argument("--rescan-50pct", action="store_true",
                    help="rebuild the 50%% export's page aggregates (~2 min each)")
    args = ap.parse_args()

    unavailable = top500.available()
    if unavailable:
        print(unavailable, file=sys.stderr)
        return 1

    panel = crawl_panel()
    touched = [p for p in panel if p["page"].n_blocked_tracking > 0]
    direct = sum(p["direct"] for p in touched)
    casc = sum(p["casc"] for p in touched)
    measured = top500.measure([p["page"] for p in touched])

    print(f"crawl: {len(touched)} pages with tracking blocks, "
          f"{len(panel) - len(touched)} without; "
          f"{sum(len(p['hosts']) for p in touched):,} blocked requests")
    print(f"       estimated directly {direct/1e6:.2f} MB, of which "
          f"{casc/1e6:.2f} MB is {'/'.join(c.lower() for c in CASCADING)}")
    print()

    print("HTTP ARCHIVE 1% -- the request tree, uniformly sampled")
    tree = tree_shape()
    if tree is None:
        print("  exports not on disk, skipped")
    else:
        print(f"  {tree['n']:,} tracker requests over {tree['pages']:,} page "
              f"domains, {tree['total']/1e9:.1f} GB")
        print(f"  roots (parser-initiated) {tree['root']/1e9:.2f} GB, "
              f"descendants (script-initiated) {tree['desc']/1e9:.2f} GB")
        print(f"  descendants per root byte              {tree['factor']:.2f}"
              f"  <- UPPER bound: a descendant of a first-party script is "
              f"refused directly, not pruned")
        print(f"  per script/document root byte          "
              f"{tree['factor_casc']:.2f}  <- the same bound, over the "
              f"context this array is applied to")
        print(f"  the same descendants per root *request*, which is the "
              f"unit the array ships in:\n    "
              f"{tree['per_root_request']/1e3:.0f} KB over all "
              f"{tree['root_n']/1e6:.1f}M roots, "
              f"{tree['per_casc_root_request']/1e3:.0f} KB over the "
              f"{tree['casc_root_n']/1e3:.0f}k script roots alone")
        print("  descendants are: " + ", ".join(
            f"{t} {100*s:.1f}%" for t, s in tree["mix"][:4]))
        print(f"  that mix against CPU_MS_PER_KIB        "
              f"{tree['cpu_ms_per_kib']:.2f} ms/KiB")
    print()

    print("PAIRED CRAWL -- the counterfactual, on 271 pages")
    n_casc = sum(len(p["casc_sizes"]) for p in touched)
    lo, hi = bootstrap_per_request_ci(panel)
    print(f"  page delta, drift-corrected          "
          f"{bytes_per_request_from_delta(panel)/1e3:6.1f} KB per request"
          f"  95% [{lo/1e3:.0f}, {hi/1e3:.0f}]")
    print(f"  page delta, raw                      "
          f"{bytes_per_request_from_delta(panel, corrected=False)/1e3:6.1f} KB "
          f"per request")
    slo, shi = bootstrap_per_request_ci(panel, stable=True)
    n_unstable = len(touched) - len(_touched(panel, stable=True))
    print(f"  the same over the pages whose loads repeat, which is the only "
          f"form of this\n  instrument worth reading -- see "
          f"`bytes_per_request_from_delta`:")
    print(f"  page delta, drift-corrected, stable  "
          f"{bytes_per_request_from_delta(panel, stable=True)/1e3:6.1f} KB per "
          f"request  95% [{slo/1e3:.0f}, {shi/1e3:.0f}]   "
          f"({n_unstable} pages dropped)")
    print(f"  page delta, raw, stable              "
          f"{bytes_per_request_from_delta(panel, corrected=False, stable=True)/1e3:6.1f}"
          f" KB per request")
    print(f"  the same, as the retired per-byte factor: "
          f"{factor_from_delta(panel):.2f} corrected, "
          f"{factor_from_delta(panel, corrected=False):.2f} raw, over "
          f"{n_casc} cascading requests")
    if not args.skip_request_set:
        extra = request_set_difference()
        quiet = [p for p in panel if p["page"].n_blocked_tracking == 0]
        # The tracker-hosts-only version of this is the split, and is
        # reported with the rest of the split below rather than twice here.
        gross = sum(extra.get(p["page"].idx, (0, 0))[0] for p in touched)
        churn = st.fmean([extra.get(p["page"].idx, (0, 0))[0] for p in quiet])
        print(f"  request set, all hosts                 "
              f"{(gross - churn * len(touched)) / casc:.2f}"
              f"  <- LOWER bound: misses any descendant both arms happened "
              f"to request")
    print()

    print("THE SPLIT -- how much of the cascade lands back on the list")
    quiet = [p["page"] for p in panel if p["page"].n_blocked_tracking == 0]
    listed_raw = bytes_per_request_from_delta(panel, corrected=False,
                                              listed=True)
    print(f"  listed only, page delta, raw         {listed_raw/1e3:6.1f} KB "
          f"per request")
    print(f"  listed only, page delta, corrected   "
          f"{bytes_per_request_from_delta(panel, listed=True)/1e3:6.1f} KB per "
          f"request")
    print(f"  listed only, corrected, stable       "
          f"{bytes_per_request_from_delta(panel, listed=True, stable=True)/1e3:6.1f}"
          f" KB per request  <- the control: the stability filter is supposed "
          f"to leave\n                                                       "
          f"       this alone, and does")
    if not args.skip_request_set:
        gross = sum(extra.get(p["page"].idx, (0, 0))[1] for p in touched)
        churn = st.fmean([extra.get(p.idx, (0, 0))[1] for p in quiet])
        print(f"  listed only, request set             "
              f"{(gross - churn * len(touched)) / n_casc / 1e3:6.1f} KB per "
              f"request  <- the same quantity, from the request sets rather "
              f"than the byte totals")
    for label, sel in (("whole page", lambda p: p.bytes_saved),
                       ("listed requests only", lambda p: p.bytes_saved_tracking)):
        sd = st.stdev([sel(p) for p in quiet])
        gross = sum(sel(p["page"]) for p in touched)
        print(f"  churn on the {label:22s} {sd/1e6:.2f} MB per page, so "
              f"{sd * len(touched) ** 0.5 / gross:5.1%} of a {gross/1e6:.0f} MB "
              f"denominator")
    print(f"  => the listed part is measured three ways inside six percent. "
          f"The whole used to be\n     unmeasurable here, because the only "
          f"instrument that could see it returned a value\n     below a "
          f"subset of itself -- 26.1 KB against the listed 47.0. That was "
          f"six pages:\n     over the ones whose loads repeat it reads "
          f"47.6 KB and the ordering holds. The two\n     now agree, which "
          f"is not the same as having found the unlisted half -- the gap "
          f"is\n     1.5 KB with an interval tens of KB wide. The shipped "
          f"figure is still the listed one.")
    print()

    print("THE UNLISTED HALF, MEASURED DIRECTLY RATHER THAN BY SUBTRACTION")
    m = top500.measure([p["page"] for p in touched])
    print(f"  unlisted third-party delta           "
          f"{m.bytes_saved_unlisted_tp/1e6:6.2f} MB over the crawl, "
          f"{m.bytes_saved_unlisted_tp/n_casc/1e3:5.1f} KB per request")
    print(f"  (no direct term: a blocked request is on the list by "
          f"construction, so none of its own bytes are in this column)")
    unl_sd = st.stdev([p.bytes_saved_unlisted_tp for p in quiet])
    unl_gross = sum(p["page"].bytes_saved_unlisted_tp for p in touched)
    print(f"  churn on it                          {unl_sd/1e6:.2f} MB per "
          f"page, against a {unl_gross/1e6:.0f} MB denominator")
    print(f"  => `bytes_saved_unlisted_tp` counts every third-party request "
          f"the Disconnect list does\n     *not* name, which is exactly the "
          f"part the shipped figure is a lower bound because it\n     "
          f"cannot see -- ad creatives, iframes, vendor CDNs -- and none of "
          f"the first-party video\n     that makes the whole-page delta "
          f"unreadable. It does not work either. Across the ten\n     "
          f"passes it reads -3.02 MB a pass with a 725% one-pass standard "
          f"deviation and a 229%\n     pooled standard error, swinging from "
          f"-34.8 to +35.6 MB; the third-party video CDNs\n     it still "
          f"contains are enough on their own. So the unlisted half remains "
          f"unmeasured\n     by this crawl, and the shipped figure remains a "
          f"lower bound with no number on the gap.")
    print()

    print("A PER-PAGE INTERCEPT -- raised by `src/live_ablation.py`, not settled")
    for listed in (True, False):
        r = intercept_fit(panel, listed=listed)
        d = r["corrected"]
        tot = d["page_mass"] + d["request_mass"]
        print(f"  {'listed-only' if listed else 'whole-page':12s} "
              f"intercept {d['intercept']/1e3:7.1f} KB/page "
              f"[{d['intercept_ci'][0]/1e3:5.0f},{d['intercept_ci'][1]/1e3:5.0f}]"
              f"  per request {d['per_request']/1e3:6.1f} KB "
              f"[{d['per_request_ci'][0]/1e3:5.1f},{d['per_request_ci'][1]/1e3:5.1f}]"
              f"  page mass {d['page_mass']/tot:4.0%}")
        print(f"  {'':12s} raw (no drift correction): intercept "
              f"{r['raw']['intercept']/1e3:7.1f} KB/page, per request "
              f"{r['raw']['per_request']/1e3:6.1f} KB")
        g = grade_intercept(panel, listed=listed, n=3000)
        print(f"  {'':12s} head to head at fixed total: {g['win_page']:.0%} "
              f"of paired bootstraps on per-page error, {g['win_cat']:.0%} on "
              f"category totals")
    print(f"  => single-host ablation reads the listed cascade at 14.6 and "
          f"14.9 KB a request over two\n     125-page runs, against this "
          f"crawl's 47.0, and both are stable. Allowing a page-level\n     "
          f"intercept moves the crawl's per-request term to 33.2 KB, in the "
          f"right direction and a\n     third of the way -- but the "
          f"intercept's interval contains zero, the two-term form is\n     "
          f"under this file's noise bar on per-page error and loses on "
          f"category totals, and on\n     the loud instrument the intercept "
          f"is zero. The crawl cannot separate the two terms,\n     which is "
          f"what `grade_saturation` already said. Nothing changes.")
    print()

    print("CONSENT MANAGERS -- raised by `src/live_ablation.py`, settled here")
    cmp_cascade(panel)
    print(f"  => the Disconnect list names OneTrust, Osano and Sourcepoint, "
          f"and single-host ablation\n     on a live page shows they are "
          f"gates rather than subtrees: +3.6 MB on braze.com,\n     "
          f"-634 KB on ted.com, reproducible to a few kB and not the same "
          f"sign. It does not reach\n     this estimator. Firefox's ETP "
          f"tracking tables are not the Disconnect list, and across\n     "
          f"all 18,891 blocked requests in the ten-pass crawl it blocked a "
          f"consent manager zero\n     times -- so `followup_bytes_for` is "
          f"never asked about one, and there is nothing to fix.\n     What "
          f"it does mean is that a candidate picked by `is_tracker` is not a "
          f"candidate ETP\n     would have blocked; see `recon` in "
          f"src/live_ablation.py.")
    print()

    print("THE FORM -- per request or per byte, which the crawl is emphatic about")
    for listed in (True, False):
        name = INSTRUMENT["tracking" if listed else "page"]
        for within in (False, True):
            f = joint_fit(panel, listed=listed, within_category=within)
            print(f"  joint fit against the {name}"
                  f"{', category demeaned' if within else''}")
            print(f"    per cascading request {f['per_request']/1e3:8.1f} KB  "
                  f"95% [{f['per_request_ci'][0]/1e3:6.1f},"
                  f"{f['per_request_ci'][1]/1e3:6.1f}]")
            print(f"    per cascading byte    {f['per_byte']:8.2f}     "
                  f"95% [{f['per_byte_ci'][0]:6.2f},{f['per_byte_ci'][1]:6.2f}]")
    buckets, reversed_order = by_request_size(panel)
    print(f"  implied per-byte factor by the blocked request's own size "
          f"(listed-only delta):")
    for (label, (b, beta, lo_, hi_)) in zip(("<10 KB", "10-50 KB", ">50 KB"),
                                            buckets):
        print(f"    {label:9s} {b/1e6:5.2f} MB of base  factor {beta:5.2f}  "
              f"95% [{lo_:5.2f},{hi_:5.2f}]")
    print(f"    a proportional cascade says these are one number; the chance "
          f"the ordering is\n    the other way round is {reversed_order:.0%}")
    for listed in (True, False):
        g = grade_form(panel, listed=listed)
        print(f"  head to head against the "
              f"{INSTRUMENT['tracking' if listed else 'page']}, same total:")
        print(f"    per-page |error|   per byte {g['page_per_byte']/1e6:6.1f} MB"
              f"   per request {g['page_per_request']/1e6:6.1f} MB"
              f"   per request wins {g['win_page']:.0%}")
        print(f"    category |error|   per byte {g['cat_per_byte']/1e6:6.1f} MB"
              f"   per request {g['cat_per_request']/1e6:6.1f} MB"
              f"   per request wins {g['win_cat']:.0%}")
    print()

    print("SHIPPED -- per distinct tracker host, not per blocked request")
    by_pass = per_pass_cascade_hosts()
    if by_pass is None:
        print("  needs the per-pass tables under paired/rep*/, skipped")
    else:
        h = host_dedup(panel, by_pass, listed=True)
        print(f"  {h['n_requests']:.0f} cascading blocks fall on "
              f"{h['n_hosts']:.0f} distinct host-page-load triples, so "
              f"{100 * (1 - h['n_hosts'] / h['n_requests']):.0f}% are repeats")
        print(f"  the same measured cascade over the two denominators: "
              f"{h['per_request']/1e3:.1f} KB per request "
              f"(FOLLOWUP_BYTES_PER_REQUEST),\n"
              f"  {h['per_host']/1e3:.1f} KB per distinct host "
              f"(FOLLOWUP_BYTES_PER_HOST)")
        for listed in (True, False):
            g = host_dedup(panel, by_pass, listed=listed)
            name = INSTRUMENT["tracking" if listed else "page"]
            print(f"  against the {name}, same total:")
            print(f"    per-page |error|   per request {g['page_flat']/1e6:6.1f} MB"
                  f"   per host {g['page_host']/1e6:6.1f} MB"
                  f"   per host wins {g['win_page']:.0%}")
            print(f"    category |error|   per request {g['cat_flat']/1e6:6.1f} MB"
                  f"   per host {g['cat_host']/1e6:6.1f} MB"
                  f"   per host wins {g['win_cat']:.0%}")
        shuffles = [host_dedup(panel, by_pass, listed=True, n=600, seed=t,
                               permute=True) for t in range(100)]
        wins = [g["win_page"] for g in shuffles]
        print(f"  control -- same counts, hosts drawn at random from the "
              f"crawl-wide pool:\n    per-page wins "
              f"{st.fmean(wins):.0%} on average over 100 shuffles, "
              f"{max(wins):.0%} at best, against {h['win_page']:.0%} real; "
              f"{sum(w >= h['win_page'] for w in wins)} of 100 reach it"
              f"\n    and their per-page error averages "
              f"{st.fmean(g['page_host'] for g in shuffles)/1e6:.1f} MB "
              f"against the flat model's {h['page_flat']/1e6:.1f}")
        print("  => the effect is which hosts, not how many requests, and it "
              "is the only\n     refinement in this file that ships.")
        print("  the alternatives to it, and the rejected shapes re-graded "
              "against it rather\n  than against the flat per-request form "
              "they were rejected on:")
        on_host_baseline(panel, by_pass)
    print()

    print("NOT SHIPPED -- saturation: do two blocked loaders prune two subtrees?")
    print(f"  {'blocks/page':12s} {'pages':>5s} {'per pass':>9s} "
          f"{'KB per request':>15s}   (buckets are per-pass counts)")
    for label, pages_n, reqs, per in saturation(panel):
        print(f"  {label:12s} {pages_n:5d} {reqs:9d} {per/1e3:15.1f}")
    print("  head to head with the flat count, same total -- which the bucket "
          "table cannot\n  do for itself, because a page-level intercept "
          "divided by k also falls in k:")
    for listed in (True, False):
        name = INSTRUMENT["tracking" if listed else "page"]
        for gamma in (0.70, 0.85, 0.95):
            g = grade_saturation(panel, listed=listed, gamma=gamma)
            print(f"    gamma {gamma:.2f} against the {name:<22s} "
                  f"per-page wins {g['win_page']:>4.0%}   "
                  f"category wins {g['win_cat']:>4.0%}")
    print("  => rejected rather than left open: the quiet instrument prefers "
          "the flat count\n     at every exponent, and only approaches a tie "
          "by approaching the flat count.")
    print()

    print("NOT SHIPPED -- a cascade that scales with the blocked request's own size")
    print("  `grade_form` tests exponents 0 and 1 on that size and 0 wins. "
          "This tests between\n  them, where the crawl does have a signal: "
          "per request, the quiet instrument reads\n  28.0, 24.4, 54.6 and "
          "51.4 KB over <5, 5-20, 20-60 and 60+ KB blocked requests.")
    for listed in (True, False):
        name = INSTRUMENT["tracking" if listed else "page"]
        for delta in (0.15, 0.30, 0.60):
            g = grade_by_size(panel, listed=listed, delta=delta)
            print(f"    delta {delta:.2f} against the {name:<22s} "
                  f"per-page wins {g['win_page']:>4.0%}   "
                  f"category wins {g['win_cat']:>4.0%}")
    print("  => rejected: a large blocked script cascades about twice what a "
          "small one does,\n     not twenty times, and a factor of two is "
          "under this crawl's resolution.")
    print()

    print("NOT SHIPPED -- a factor per page category")
    per_category(panel)
    print()

    print("NOT SHIPPED -- a figure per tracker category, from the Disconnect list")
    by_category = crawl_categories(panel)
    if by_category is None:
        print("  the `disconnect` extension is not importable, skipped")
    else:
        shape = None
        if not args.skip_50pct:
            if args.rescan_50pct or not HA50_CATEGORY_CACHE.exists():
                if list(HA50_GLOB.parent.glob("*.parquet")):
                    print(f"  building {HA50_CATEGORY_CACHE.name} ...")
                    build_ha50_category_cache()
            if HA50_CATEGORY_CACHE.exists():
                shape, note = category_shape_from_ha50()
                print(f"  the 50% export's own shape: {note}")
        per_category_cascade(panel, by_category, ha50_shape=shape)
    print()

    print("NOT SHIPPED -- a figure per tracker role, an axis tests/top500.py "
          "bounds size on")
    by_role = crawl_roles(panel)
    if by_role is None:
        print("  tests/top500.py is not importable, skipped")
    else:
        per_category_cascade(panel, by_role)
    print()

    print("SHIPPED, WEAKLY -- a factor per match level, the rung of the "
          "table that answered")
    by_rung = crawl_match_levels(panel)
    if by_rung is None:
        print("  llm_classifier is not importable, skipped")
    else:
        rung_cascade_shape(panel, by_rung,
                           extra=None if args.skip_request_set else extra)
    print()

    print("NOT SHIPPED -- a factor per tracker site")
    share = per_host_parser_share()
    if share:
        known = [(h, b) for p in touched for h, b, c in p["hosts"]
                 if c and h in share]
        covered = sum(b for _, b in known)
        mean = sum(share[h] * b for h, b in known) / covered
        cov = covered / max(sum(b for p in touched
                                for _, b, c in p["hosts"] if c), 1)
        def by_rootness(h, b, c):
            return (share.get(h, mean) / mean) * b if c else 0.0

        for target in ("page", "tracking"):
            grade_shape(panel, by_rootness,
                        f"weighted by how often the markup asks for the host "
                        f"directly, against the {INSTRUMENT[target]}\n    "
                        f"(1% exports; {cov:.0%} of cascading bytes covered)",
                        target=target)
    if not args.skip_50pct:
        for cache, build in ((HA50_CACHE, build_ha50_cache),
                             (HA50_MASS_CACHE, build_ha50_mass_cache)):
            if args.rescan_50pct or not cache.exists():
                if not list(HA50_GLOB.parent.glob("*.parquet")):
                    print("  50% export not on disk, skipped")
                    break
                print(f"  building {cache.name} ...")
                build()
        if HA50_CACHE.exists():
            per_req, note = per_host_cascade_bytes()
            hit = sum(1 for p in touched for h, _b, _c in p["hosts"]
                      if h in per_req)
            tot = sum(len(p["hosts"]) for p in touched)
            for target in ("page", "tracking"):
                grade_shape(panel, lambda h, b, c: per_req.get(h, 0.0),
                            f"by each host's fitted share of a page's tracker "
                            f"mass, against the {INSTRUMENT[target]}\n    "
                            f"(50% export; {hit}/{tot} blocked requests "
                            f"covered; {note})",
                            target=target)
        if HA50_MASS_CACHE.exists():
            _split, note = per_host_cascade_split()
            print(f"  per host, split listed against unlisted: {note}")
    print()

    followups = SHIPPED_BYTES_PER_REQUEST * n_casc
    # What is left of the measured CPU saving once the blocked requests
    # themselves are paid for, spread over the follow-up bytes. Computed from
    # the panel rather than written down, because a crawl of a different size
    # -- ten pooled passes rather than one -- changes the direct total and a
    # literal here would quietly rescale the cap by ten.
    direct_cpu_s = sum(p["direct_cpu_s"] for p in touched)
    cpu_cap = (((measured.cpu_saved_s - direct_cpu_s) * 1000.0)
               / (followups / 1024.0))
    print("SHIPPED")
    print(f"  FOLLOWUP_BYTES_PER_REQUEST = "
          f"{SHIPPED_BYTES_PER_REQUEST // 1000} KB for SCRIPT and HTML, "
          f"0 elsewhere  (retired: {RETIRED_FACTOR_PER_BYTE} per byte)")
    print(f"  FOLLOWUP_CPU_MS_PER_KIB = 0.9   (the descendant mix says "
          f"{tree['cpu_ms_per_kib']:.1f}; the crawl's {measured.cpu_saved_s:.0f} s "
          f"less {direct_cpu_s:.0f} s of direct estimate leaves {cpu_cap:.2f})"
          if tree else
          f"  FOLLOWUP_CPU_MS_PER_KIB = 0.9   (crawl cap {cpu_cap:.2f})")
    print(f"  giving {(direct + followups)/1e6:.1f} MB against "
          f"{measured.bytes_saved/1e6:.1f} MB measured "
          f"({(direct + followups)/measured.bytes_saved:.2f}x) and "
          f"{measured.bytes_saved_raw/1e6:.1f} MB raw "
          f"({(direct + followups)/measured.bytes_saved_raw:.2f}x)")
    # The same ratio over the pages the whole-page delta can measure, which is
    # the one `tests/top500.py` now bounds; the 1.48x above divides by a
    # denominator six churning pages take 38% off.
    stable_pages = _touched(panel, stable=True)
    stable_raw = sum(p["page"].bytes_saved for p in stable_pages)
    stable_est = sum(p["direct"] + SHIPPED_BYTES_PER_REQUEST
                     * len(p["casc_sizes"]) for p in stable_pages)
    print(f"  over the {len(stable_pages)} pages whose loads repeat: "
          f"{stable_est/1e6:.1f} MB against {stable_raw/1e6:.1f} MB raw "
          f"({stable_est/stable_raw:.2f}x)  <- `Bounds.bytes_page_ratio_max_stable`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
