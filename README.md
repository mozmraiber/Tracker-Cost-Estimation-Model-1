# Performance Cost Regression for Firefox-Blocked Tracker Requests

Predicting the transfer size of tracker requests blocked by Firefox Enhanced Tracking Protection, using pre-response features available at block time.

## Problem

When Firefox blocks a tracker request, the response is never observed. The browser sees the URL, resource type, and request metadata, but not the response size. This project trains models to predict what the response would have been, enabling the privacy dashboard to show users concrete savings: "Firefox saved you approximately 2.3MB of bandwidth this week."

A domain-level lookup table is insufficient because the same domain serves vastly different resources (`googletagmanager.com/gtag/js` = 93KB script, `googletagmanager.com/collect` = 0-byte beacon). A path-level table is infeasible (50M entries, 1.1GB, immediately stale).

## Results

- **XGBoost with Tweedie loss**: 39.4% MAE improvement over lookup table baseline
- **Character-level URL CNN**: best ranking quality (Spearman 0.977)
- **Weekly aggregate accuracy**: within 10% of true total 63% of the time (vs 15% for LUT)
- 10 model architectures compared; loss function selection matters more than architecture

## Structure

```
sql/                    BigQuery queries for HTTP Archive data extraction
src/model/
  train_per_request.py  Main pipeline: LUT, Ridge, RF, XGBoost, LightGBM, CatBoost
  train_advanced.py     Tweedie loss, two-stage, URL hashing, quantile regression
  advanced_analysis.py  Calibration, aggregation simulation, distribution shift
  url_cnn.py            Character-level CNN for learned URL representations
data/
  raw/                  HTTP Archive request data
  external/             Disconnect list
models/per_request/     Trained models and results
output/                 Visualizations
duckdb-disconnect/      DuckDB extension: is_tracker(url) over the Disconnect list
llm-classifier/         Shipping estimator: a compiled-in hierarchy of conditional
                        means, as a plain Rust library with no Python in it
llm-classifier-python/  The CPython extension over it, and the only crate with pyo3
llm-classifier-baseline/  Empty extension, for measuring the above's size
xgb-classifier/         The same interface over the XGBoost artifact, via m2cgen
tests/                  Browsing-journey aggregate-error tests (pytest)
```

### Two estimators behind one interface

`llm-classifier-python` and `xgb-classifier` are Python extensions answering
the same `estimate_resources(url, context, initiator, method) -> (bytes,
cpu_ms)`. They differ in where the byte estimate comes from: the first calls
the `llm-classifier` library, which looks the request up in a compiled-in
hierarchy of conditional means; the second runs
`models/per_request/xgb_transfer_bytes.json` — the `xgboost_shipped` artifact
this paper reports — transpiled to Rust by
[m2cgen](https://github.com/BayesWitnesses/m2cgen) and fed by a port of the
`engineer_features` pipeline.

They also put the last two arguments to very different use, which is the same
story as the size comparison below: the table has the URL path and the booster
does not. `xgb-classifier` one-hots both into its feature vector, and they are
worth 19 points of journey error to it. `llm-classifier` uses the method for
one thing — a HEAD or a CORS preflight cannot carry a response body, so it is
answered from the method rather than from the URL — and does not key on the
initiator at all, because the path already separates the 17.2 KB
script-initiated GETs from the 3.5 KB parser-discovered ones. It also defaults
neither argument, where `xgb-classifier` defaults both, and has no JSON request
context, which that crate added because its model was fitted on one.

`llm-classifier` takes one argument `xgb-classifier` has no answer for:
`include_followups`, defaulting to `False`, which asks for the cost of the
requests the blocked request would itself have made — the cascade the paper's
limitations section names and declines to quantify. Left at its default the two
extensions are interchangeable and the comparison below is unaffected. Set, the
estimate goes from 25.0 MB to 58.3 MB a pass over the paired crawl's 1,875
blocks, against 55.9 MB measured over Disconnect-matched requests and 39.3 MB
over the whole page — two figures that ought to be the other way round, and
the reason the cascade is known less well than a single number suggests.

It is charged **per blocked request, not per byte of it**, and that is the one
thing the crawl is emphatic about. Fit both terms at once and the size term
vanishes: 30.1 KB of subtree per cascading request (95% [10.5, 42.8]) against
0.05 bytes per cascading byte (95% [0.00, 0.53]), and the same with each
page's category demeaned out. Cut the blocked requests by their own size and
the implied per-byte factor falls monotonically — 3.79 under 10 KB, 2.63 from
10 to 50 KB, 0.37 above 50 KB, with a 22% bootstrap chance the ordering is the
other way. Held to the same crawl-wide total so neither form can win by being
bigger, per request beats per byte on the quiet Disconnect-only delta (86% of
paired bootstraps on per-page error, 65% on category totals) and loses on the
loud whole-page one (10% and 68%). It replaced a proportional 2.3 bytes per
byte in 2026-09, and because HTTP Archive's tracker scripts average 31.8 KB —
which that factor charged 73 KB — the aggregate barely moved; what changed is
that a 2 KB `gtag/js` stub is no longer charged 4.5 KB of subtree and a 106 KB
`fbevents.js` is no longer charged 244 KB.

The magnitude is where repeating the crawl changed the answer. The paired
crawl is the only counterfactual — one arm blocked, one did not — and it
offers the quantity two ways. Its **whole-page** delta is everything the
blocking arm did not fetch, wherever it came from, and is what the shipped
70 KB was set from: on one pass it read 70.2 KB a request. Ten pooled passes
read 26.1 KB, 95% [-26, 70]. Its **listed-only** delta, the same difference
counted over Disconnect-matched requests, reads 43.7 KB raw, 47.0 KB
drift-corrected, and 49.5 KB from the request sets rather than the byte
totals.

Those cannot both be measurements: the second counts a subset of what the
first counts and comes out larger. The whole-page instrument is the one that
is failing, and the passes say why — its total has a standard deviation of
74% per pass against the listed-only delta's 16.2%, so pooled it carries
23.4% against 5.1%. So the shipped figure is now **47 KB**, the listed-only
one, and its honest label is a lower bound: it is the part of a blocked
tracker's subtree that lands back on hosts some list names, and the ad
creatives and iframes on hosts no list names are real and are not in it.

HTTP Archive brackets the same quantity from above. Its 1% URL exports carry
`initiator_type`, so over 9.7M tracker requests on 3.2M pages the tree splits
into roots (parser-initiated, 12.6 GB) and descendants (script-initiated,
58.5 GB) — 62 KB per script root, an *upper* bound because the export says a
script asked for the request and not which script: a tracker's child is
pruned, a first-party script's child is refused directly. Its numerator counts
tracker requests, so what it bounds is exactly the listed part. 47 KB measured
against ≤62 KB bounded, from two crawls, two browsers and two months apart.

The cost of calibrating against the crawl's quiet denominator is that the
sharp page-level bound in [tests/top500.py](tests/top500.py), which divides by
that denominator, is now a calibration check rather than an independent one.
It reads 1.04 where it used to read 1.47 and derive that centre from the gap
between 70 and 36. See `Bounds.bytes_tracking_ratio_min`.

Nine refinements have been measured and one survived. The one that ships is
`FOLLOWUP_HOST_SCALE`, a per-host *shape* for the cascade, matched by hostname
suffix so that one `google-analytics.com` entry prices every regional shard
and one `sentry.io` entry prices every per-customer subdomain, with exact
entries kept only where a child disagrees with its suffix — `doubleclick.net`
needs three, because `securepubads` cascades far more than `ad` or
`googleads`. It comes from single-host ablation on live pages
(`src/live_ablation.py`), 646 measurements over four runs and 115 hosts, and
it is levelled and clamped so it redistributes the cascade without changing
the crawl-wide total. Held out, it predicts a host's cascade to 40.2 KB
against the single constant's 55.7. The keys are FNV-1a/32 hashes and the
scales are single bytes, so the whole table is 460 bytes and no hostname is in
the binary.

The eight that did not survive: a figure per *tracker* category from the
Disconnect list, a factor per *page* category, two per tracker site (by how
often the markup asks for a host directly, and by each host's fitted share of
a page's tracker mass — 1,826 hosts fitted over 510k pages from the 50%
export), a cascade scaling with the blocked request's own size, saturation in
the number of loaders blocked on a page, a page-level intercept, and the
unlisted half of the cascade measured directly from the crawl.

The tracker-category one is the most tempting and the most instructive. The
crawl puts Advertising at 36.7 KB a request and Analytics at 27.5 KB against
a single 38.4 KB, and cannot separate them: every 95% interval contains the
single figure, and Social — the one the story is most confident about — comes
out highest of the three at 76 KB. That is an instrument-resolution problem
rather than a modelling one, and the arithmetic generalises to any
categorisation: 7,089 cascading blocks over 271 pages support one figure to
about a third of itself, so a three-way split widens each interval by roughly
√3 while the categories differ by less than that.

An earlier version of this section predicted that "a crawl ten times larger
would separate 39 KB from 22 KB". The crawl is now ten times larger and the
intervals are *wider*, not narrower, which is worth recording because the
prediction confused two kinds of noise. Ten passes over the same 271 pages
remove the variance in how a page load comes out and leave the variance in
which pages were crawled, and the bootstrap here resamples pages. Repetition
bought the page-level denominators a great deal; a per-category split needs a
crawl ten times *wider*, which is a different experiment.

The per-site pair used to be recorded as coin flips; graded against the quiet
Disconnect-only delta rather than the loud whole-page one they are worse than
a single constant, each winning 2% of paired bootstraps on per-page error and
staying a coin flip on category totals (41% and 43%). HTTP
Archive can identify what co-occurs with a tracker inside the list and cannot
identify what a tracker causes outside it: fit against unlisted third-party
mass and www.google-analytics.com, a 20 KB beacon, is credited with 255 KB of
it — and the same collinearity sinks the category version, where Analytics,
being on almost every page, doubles as the intercept and takes the whole
cascade while Advertising is left with none. Most of what all four refinements
were reaching for turned out to be the functional form: once the cascade is
charged per request, neither a per-site nor a per-category shape adds anything
measurable. See `FOLLOWUP_BYTES_PER_REQUEST` in
[llm-classifier/src/lib.rs](llm-classifier/src/lib.rs) and
[llm-classifier/scripts/fit_followups.py](llm-classifier/scripts/fit_followups.py),
which re-runs all of it in about a minute once the 50% export's two per-page
aggregates are cached.

With those supplied it scores exactly what `xgboost_shipped` scores, so the
comparison is the model against the table and nothing else. The table wins by
277x on size and 1.6-4.6x on journey error; see
[xgb-classifier/README.md](xgb-classifier/README.md).

## Tests

```
pytest                  # tests/, the browsing-journey error suite
```

`tests/` checks the property the per-request metrics do not capture: that a
*sum* of predictions over a correlated sample stays close to the truth, since
the dashboard reports a weekly total rather than individual requests. Journeys
are sequences of page visits sampled from the HTTP Archive log, so each visit
contributes that page's whole set of blocked tracker requests and the
within-page correlation is preserved.

Every candidate solution — the domain and domain+type lookup tables, the
smoothed path table, XGBoost trained in-test, and the shipped XGBoost artifact
— has a median journey-error budget it must stay inside, currently met with
roughly 2x margin. Two deliberately wrong predictors are included as a
negative control, one biased low and one high, so a bug that made the metric
insensitive to prediction quality would show up as those tests passing.

The suite reads `data/raw/per_request_1pct.csv` and `models/per_request/`, both
gitignored; tests skip with an explanatory message when they are absent.

`tests/test_top500_estimates.py` is the out-of-sample half: it scores the same
estimator against the paired crawl below, where Firefox chose what to block.
Because a single aggregate over 271 pages can be carried by a handful of them,
it also cuts the crawl up — by page category
(`data/tranco_500_categories.csv`, seven hand-labelled kinds of site), over
200 seeded random half- and quarter-crawls, and by leaving each page out in
turn. The categories have already caught two biases the grand total could not
see: news and media pages at -18.7% when the aggregate read -7.1%, and
`infrastructure` at +28.1% today. Mis-scaled predictors are run against the
subset bounds too, so a loose bound that nothing can fail shows up as a
failing negative control.

The page-level half of that suite divides by the Disconnect-matched delta
wherever it can, because that instrument is four times quieter than the whole
page (0.18 MB of crawl churn per page against 1.37 MB) and the extra power
buys finer statements rather than a tighter headline: the ratio is bounded per
page category on the two categories whose churn is small enough to grade
(news at 5.7% standard error, productivity at 9.9%, carrying 57 MB of the 78
MB predicted), on *every* one of the 200 random half-crawls rather than on
their median, and against dropping any single page, where no page moves it by
more than 3.3%. A composition error is the negative control for that:
shifting 30% of the estimate from productivity to news leaves the crawl-wide
ratio at 1.47 and takes both categories outside the tolerance around their
recorded ratios, which is the error a single aggregate cannot see.

## Paired tracker-blocking crawl

`src/firefox_crawl_500_tracking.py` loads the same page twice in headless
Firefox — once with tracking-content blocking off, once with it on, as a
Private Window behaves — and records per page, per arm: every request URL with
its `_transferSize`, every request the URL classifier refused (with the
nsresult), and CPU seconds for the whole Firefox process tree. It measures
what blocking actually saves, rather than predicting it.

```
python src/build_tranco_500_top.py --tranco data/external/tranco_top1m.csv \
    --out data/tranco_500_top.txt --target 500     # top 500 reachable pages
python src/patch_playwright_firefox.py --apply      # once; see below
python src/firefox_crawl_500_tracking.py --verify-etp

python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
    --mode normal  --out data/raw/firefox_crawl_500_tracking/normal
python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
    --mode private --out data/raw/firefox_crawl_500_tracking/private

python src/compare_tracking_arms.py \
    --normal  data/raw/firefox_crawl_500_tracking/normal \
    --private data/raw/firefox_crawl_500_tracking/private \
    --out     data/raw/firefox_crawl_500_tracking
```

### Repeat visits

One load of a page is a noisy measurement of it: ad auctions fill differently
on every impression, lazy content varies, and CPU carries whatever else the
machine was doing. `--repeats N` visits the whole list N times so a page can
be averaged over visits and its spread inspected:

```
python src/firefox_crawl_500_tracking.py --urls data/tranco_500_top.txt \
    --repeats 10 --mode normal \
    --out data/raw/firefox_crawl_500_tracking_x10/normal
```

Each round is a full pass over the list and only starts once the previous one
has finished, so the two visits to a site are an entire pass apart — hours,
for 500 URLs — rather than back to back. The URL order is deliberately held
fixed across rounds: any reshuffle moves some site to the end of one round and
the start of the next, which is the close pair the sequencing exists to
prevent. `--round-gap-s S` puts a floor on the period between round starts,
which is what keeps a short `--limit` list honest.

Round *k* lands in `<out>/repNN/` with the ordinary flat arm layout, so
`compare_tracking_arms.py` and the rest can be pointed at one round unchanged;
`<out>/_pages.csv` pools every round with a `rep` column, and each page's
`t_start_iso` records when that particular visit happened. A `--repeats 1` run
(the default) writes the flat layout directly under `<out>/` as before.

Visits are not independent at the site's end — it can recognise a returning IP,
and frequency caps shape which ads it serves — so read the spread across rounds
as the measurement's repeatability, not as N independent draws.

`compare_tracking_arms.py` pairs the arms into `paired_pages.csv` (bytes and
CPU saved per page) and `blocked_observed_bytes.csv`, which annotates each
blocked tracker request with the size that *same* request actually transferred
in the control arm — an in-browser ground-truth label of the quantity the
model predicts at block time.

`paired_pages.csv` carries the byte delta twice: over the whole page, and over
Disconnect-matched requests alone (`bytes_saved_tracking`). The second is the
quieter measurement by a factor of four — a page's two loads differ mostly in
first-party media, and none of that is in it — which is what lets
`tests/top500.py` bound the estimator to ±7% at page level instead of ±15%.
The cost is that it cannot see what a blocked tracker would have pulled in
from a host the list does not name, so the estimate sits at 1.5x of it rather
than at 1.0x.

**The bundled Firefox must be patched first.** Firefox fetches the ETP tracker
lists at runtime through Remote Settings, and Playwright's build hard-disables
Remote Settings in packaged JavaScript (`shouldSkipRemoteActivity` is patched
to `return true`), which no pref can override. Unpatched, ETP silently blocks
*nothing* and the blocking arm is an expensive duplicate of the control arm.
`patch_playwright_firefox.py` removes that one statement, keeps a backup, and
is reversible with `--revert`; `playwright install` silently undoes it, so
re-run `--check` after upgrading Playwright. The crawler independently refuses
to record a blocking arm whose canary page shows no real blocks.

Two caveats worth carrying into any writeup. The control arm blocks
**nothing** — tracking, cryptomining and fingerprinting protection are all off
there, unlike a real normal window — so that every request the page made has an
observed size and any request the blocking arm refused can be priced against a
real one. The delta is therefore all of ETP Standard's content blocking;
`protection=='tracking'` restricts it to tracking content, which is what the
estimator models and what every analysis here reports. And per-page CPU is
inflated unevenly by contention when `--workers > 1`; `host_cpu_pct_mean` is
recorded so analysis can control for it, and `--workers 1` gives clean CPU at
the cost of wall time.

## Scoring the estimator against real ETP

The paired crawl turns "what would this blocked tracker have cost?" into a
question with a ground-truth answer, since the control arm fetched the same
request. `src/compare_estimate_vs_etp.py` scores the shipped `llm-classifier`
table against it, feeding the table only the URL and request context — what
the browser has before a response exists.

Figures below are per pass of the ten-pass crawl. `compare_estimate_vs_etp.py`
reads one arm directory and so scores a single pass; the pooled totals quoted
here come from [tests/top500.py](tests/top500.py), which reads the tables
`src/pool_tracking_reps.py` writes.

On the requests ETP actually blocked the table is close to unbiased in
aggregate: **20.0 MB predicted vs 19.6 MB observed, +2.1%** per pass, over
11,290 priced blocks across the ten. Per request it is much weaker (MAE
8.6 kB, Spearman 0.66), which is the behaviour its own docstring claims — a
conditional mean, good for totals, unreliable per request.

Priced over *every* request ETP blocked — the unmatched ones included, since
the table needs only a URL — the estimate comes to **25.0 MB against a
39.3 MB measured transfer delta** per pass on those 271 pages (43.5 MB after
the drift correction below), so **0.64x** of what the pages actually shed.
Part of that gap is the cascade: a blocked tracker also never injects its own
subresources. With `include_followups`, which prices those, the same
population comes to **58.3 MB, or 1.34x** the drift-corrected delta.

Three further denominators check the same total, in increasing order of
independence and decreasing order of sharpness: the tracker-only delta
(1.04x, bounded to ±10%), the whole-page raw delta (1.48x, a factor-level
band only), and HTTP Archive's own crawl of the same 500 domains (0.20x,
±30%) — the only one that does not depend on this repo's crawl being sound,
and, since the cascade constant is now calibrated against the first of the
three, the only one of them that is fully independent of the estimator.

### The aggregate is cancelling, not accurate

That +2.1% is a sum over kinds of tracker that largely cancel. Splitting the
priced blocks by `family-kind` — the Disconnect list's own category for the
URL, crossed with what the request was for — gives:

| role | priced requests | observed | bias | contribution to the total |
|---|---|---|---|---|
| `ad-script` | 2,922 | 105.2 MB | −3.6% | −1.9 pts |
| `analytics-script` | 1,868 | 40.1 MB | +6.9% | +1.4 pts |
| `social-script` | 751 | 31.8 MB | +16.8% | +2.7 pts |
| `ad-beacon` | 1,821 | 8.1 MB | +41.6% | +1.7 pts |
| `fingerprinting-script` | 99 | 4.7 MB | −83.9% | −2.0 pts |
| `ad-pixel` | 1,529 | 1.8 MB | +87.8% | +0.8 pts |

Eleven points of error by role, two points in total. None of it was visible
before: a total cannot see a reallocation, and the existing cut by *page*
category cannot either, because every kind of page carries every kind of
tracker. `tests/top500.py` now bounds the contribution column directly — no
role may move the crawl's total by more than 4 points — with
`TRACKER_COMPOSITION_SKEW` as the negative control: a 15% reallocation
between the two largest script roles leaves the aggregate untouched and every
page category inside its bound, and fails the new one.

Both halves of the taxonomy are measured rather than invented, which matters
because a taxonomy drawn after looking at the errors would guarantee its own
result. A functional one — loader against leaf, which is the split that
should matter — was tried first and is not identifiable from the signals
available: the obvious proxy is rootness, the share of a host's requests the
markup asks for directly, and it puts `connect.facebook.net`, the canonical
loader, at 0.03 while `static.cloudflareinsights.com`, a beacon, scores 0.99.
Tag setups inject everything from script, so rootness measures how a vendor is
installed, not what it does.

Two of the misses are recorded rather than fixed, because they are not table
bugs. `social-script` is almost all `snap.licdn.com`, where the table answers
20.3 kB for `/li.lms-analytics/insight.min.js` and HTTP Archive backs it at
20.2 kB over 7,427 observations of that exact path — while this crawl measures
1.9 kB, 322 times, without varying. Two crawls, one URL, a factor of ten;
pulling the table to 1.9 kB would be fitting it to the test.

### A third cut: how specifically the URL was recognised

`llm_classifier.classify_url` reports which rung of the estimator's fallback
hierarchy answered — exact path, templated path, asset family, host,
templated host, extension-and-query, extension, or nothing at all — out of
the same table walk that produces the estimate. It is the cut with the least
room to argue about independence: a rung is assigned by `build_table.py` out
of HTTP Archive, before any crawl exists to grade against.

It is also the sharpest. The rungs are not equally calibrated, and the error
runs with specificity:

| rung | priced requests | observed | bias | contribution |
|---|---|---|---|---|
| `template` | 395 | 9.9 MB | +55.8% | +2.8 pts |
| `prefix` | 2,916 | 73.7 MB | +13.8% | **+5.2 pts** |
| `host_template` | 3,611 | 87.8 MB | −4.9% | −2.2 pts |
| `ext_query` | 4,087 | 20.6 MB | −24.3% | −2.6 pts |
| `ext`, `context` | 211 | 2.3 MB | −88%, −98% | −1.1 pts |

The table over-answers where it recognises the URL and under-answers where it
does not. Some of that is selection — a URL reaches `prefix` *because* it had
no exact-path entry, and coverage correlates with size — but the aggregate is
then right only by cancellation, and the mix of rungs in a real browsing
session is not the mix in this crawl. Every gradeable rung is pinned to its
measured value in `tests/top500.py`, passing or not, because unlike a page
category or a tracker role a rung's bias is `build_table.py`'s doing and a
rebuild could close it.

This is also the one refinement to the *cascade* that ships, and it ships
weakly. Fitting a cascade per rung, the two quiet instruments agree on the
order and on the magnitudes — `host_template` 27.0/32.5 KB, `prefix`
43.9/41.9, `ext_query` 75.4/80.4 — but resampling pages puts
P(`prefix` > `host_template`) at only 78% and 64%, and the loud whole-page
delta orders `ext_query` the other way. So the shape is shrunk toward the
single figure by how much of it survives its own noise (a between-rung
standard deviation of 5.9 KB against a mean standard error of 27.3 KB) and
then levelled, leaving `FOLLOWUP_RUNG_SCALE` at ±6% with a crawl-wide total
identical to the flat constant's. Levelling is what makes a weak shape safe:
if it is noise, some requests are priced 6% high and others 6% low and no
total moves. A test asserts that directly — raising every factor by 0.05
fails it at +4.9% while every ratio bound in the module still passes.

The same split by *tracker role* applied to the cascade separates nothing —
`ad-script` 34.9 KB [0, 65] against `analytics-script` 26.6 KB [0, 61] and one
figure of 38.4 KB — and the contrast with the size bounds is not about the
taxonomy. A size bound has 11,290 per-request observations; a cascade bound
has 271 pages however many requests sit on them. Splitting is affordable
exactly when the measurement is per request, which is why the rung split
above buys ±6% of shape on the cascade and several points of real bias on the
size estimate.

Estimating savings *offline* is where it goes wrong, and the size table is not
at fault. A plain `is_tracker(url)` lookup flags 4.4x more requests than
Firefox blocks, inflating predicted savings to 244.5 MB.
`src/firefox_etp_proxy.py` applies the other three tests Firefox applies —
third-party, entity allowlist, allowlist exceptions — using the lists the
browser itself consults:

```
python src/fetch_firefox_lists.py --mozilla-central ~/firefox
python src/eval_etp_proxy.py \
    --normal  data/raw/firefox_crawl_500_tracking/normal \
    --private data/raw/firefox_crawl_500_tracking/private \
    --paired  data/raw/firefox_crawl_500_tracking/paired_pages.csv \
    --out     data/raw/firefox_crawl_500_tracking
```

Graded against what Firefox really did, over the 44,131 requests that recurred
in both arms. This section is the one below that still reports the
single-pass crawl: `eval_etp_proxy.py` reads one arm directory, and what it
measures — which requests a list lookup flags — is a property of the lists
rather than of the sample, so repeating the crawl would not move it.

| classifier | precision | recall | F1 | false-positive bytes |
|---|---|---|---|---|
| `is_tracker(url)` | 0.293 | 0.999 | 0.453 | 168.2 MB |
| `FirefoxETPProxy` | 0.954 | 0.965 | 0.960 | 2.7 MB |

Same size table, corrected classification: predicted savings drop from
244.5 MB to 61.7 MB against 55.6 MB observed on those requests. The lists are
the whole difference. What a naive lookup over-counts is mostly first-party
(2,033 requests — Tranco's head contains the ad-tech vendors' own sites) and
same-entity (1,944 — `ltwebstatic.com`/Shein, `muscache.com`/Airbnb,
`ebayimg.com`/eBay).

Party must be judged against the document that *loaded*, not the one
requested: Tranco lists `onelink.me`, which 301s to `www.appsflyer.com`, and
scoring against the pre-redirect URL made AppsFlyer's own first-party assets
look third-party — 14 MB of phantom blocking, and precision 0.805 instead of
0.95 when that bug was in place. `final_page_url()` follows the HAR redirect
chain.

Two caveats. 599 of the 1,201 exceptions are `isPrivateBrowsingOnly`; the
crawl's blocking arm is a normal window with tracking protection forced on, so
it ignores them and blocks slightly more than a real Private Window would —
`FirefoxETPProxy(pbm=True)` models the latter. And the shipped `*-digest256`
lists are SHA-256 hashes that cannot be enumerated, so the proxy reads the
upstream shavar sources they are built from; the two can disagree at the
margins, which is consistent with the 54 remaining false negatives.

### Measured savings need a drift correction

The raw net page-level delta of 36.2 MB per pass is not a safe thing to
compare against. The 199 pages where ETP blocked nothing must show zero
saving in expectation, yet they total **−3.1 MB** per pass of pure
crawl-to-crawl drift (ad rotation, changed content), −15.6 kB per page. The
271 pages it did act on total 39.3 MB; netting the per-page drift out of
those leaves **43.5 MB**. Drift has run in both directions across crawls —
the single-pass crawl this replaced measured it at +12.3 MB — so the
correction matters more than its sign on any one run.

Repeating the crawl is what turned the size of that correction into a number
rather than a worry. Each pass is an independent draw, so the spread across
the ten is the sampling distribution: the whole-page delta has a standard
deviation of 74% per pass, which is 23.4% on the pooled figure, against 16.2%
and 5.1% for the same delta counted over Disconnect-matched requests only.
The CPU delta is worse still — raw it is *negative*, −89 s per pass, and only
a −6.2 s-per-page drift correction makes it positive. See
`Bounds.cpu_ratio_min` in [tests/top500.py](tests/top500.py).

One expectation the ten passes overturned: predicted was supposed to land
*below* measured, because measured includes the follow-ups a blocked tracker
never got to make and the estimator did not price them. It does not. The
Disconnect-matched delta, 55.9 MB per pass, exceeds the whole-page delta of
39.3 MB despite counting a strict subset of the same requests, which means
the blocking arm fetches **more** unlisted content than the control arm —
16.7 MB per pass more, at t = −2.1. A whole-page difference therefore does
not bound what blocking removed, and the structural ceiling that rested on
it has been retired.

## Tracker labelling

`duckdb-disconnect/` builds a DuckDB C++ extension that classifies URLs against
the Disconnect services list in-process, so labelling a crawl needs no join:

```sql
LOAD 'duckdb-disconnect/build/disconnect.duckdb_extension';
SELECT count(*) FILTER (WHERE is_tracker(url, 'Advertising,Analytics,Social,Cryptomining'))
FROM requests;
```

See [duckdb-disconnect/README.md](duckdb-disconnect/README.md) for the function
reference, matching rules and build instructions.

## Data

Training data: 348,909 tracker requests from the HTTP Archive June 2024 crawl (0.1% sample of 348M total). Features: URL path structure, file extension, resource type, initiator type, target-encoded domain statistics. Target: `transfer_bytes`.

## Paper

`performance_cost_estimation_tracker_domains.tex` (18 pages, compile with `pdflatex` + `bibtex`)
