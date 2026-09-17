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

`compare_tracking_arms.py` pairs the arms into `paired_pages.csv` (bytes and
CPU saved per page) and `blocked_observed_bytes.csv`, which annotates each
blocked tracker request with the size that *same* request actually transferred
in the control arm — an in-browser ground-truth label of the quantity the
model predicts at block time.

**The bundled Firefox must be patched first.** Firefox fetches the ETP tracker
lists at runtime through Remote Settings, and Playwright's build hard-disables
Remote Settings in packaged JavaScript (`shouldSkipRemoteActivity` is patched
to `return true`), which no pref can override. Unpatched, ETP silently blocks
*nothing* and the blocking arm is an expensive duplicate of the control arm.
`patch_playwright_firefox.py` removes that one statement, keeps a backup, and
is reversible with `--revert`; `playwright install` silently undoes it, so
re-run `--check` after upgrading Playwright. The crawler independently refuses
to record a blocking arm whose canary page shows no real blocks.

Two caveats worth carrying into any writeup. Cryptomining and fingerprinting
blocking is active in **both** arms, because that is what ETP Standard does in
a normal window, so only `protection=='tracking'` rows are savings
attributable to private browsing. And per-page CPU is inflated unevenly by
contention when `--workers > 1`; `host_cpu_pct_mean` is recorded so analysis
can control for it, and `--workers 1` gives clean CPU at the cost of wall time.

## Scoring the estimator against real ETP

The paired crawl turns "what would this blocked tracker have cost?" into a
question with a ground-truth answer, since the control arm fetched the same
request. `src/compare_estimate_vs_etp.py` scores the shipped `llm-classifier`
table against it, feeding the table only the URL and request context — what
the browser has before a response exists.

On the requests ETP actually blocked the table is close to unbiased in
aggregate: **21.3 MB predicted vs 21.8 MB observed, −2.3%**. Per request it is
much weaker (MAE 9.1 kB, Spearman 0.70), which is the behaviour its own
docstring claims — a conditional mean, good for totals, unreliable per request.

Estimating savings *offline* is where it goes wrong, and the size table is not
at fault. A plain `is_tracker(url)` lookup flags 4.4x more requests than
Firefox blocks, inflating predicted savings to 157 MB.
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

Graded against what Firefox really did, over the 44,797 requests that recurred
in both arms:

| classifier | precision | recall | F1 | false-positive bytes |
|---|---|---|---|---|
| `is_tracker(url)` | 0.285 | 0.999 | 0.443 | 221.8 MB |
| `FirefoxETPProxy` | 0.950 | 0.965 | 0.958 | 2.5 MB |

Same size table, corrected classification: predicted savings drop from 157 MB
to 51.8 MB against 54.8 MB observed on those requests. The lists are the whole
difference. What a naive lookup over-counts is mostly first-party (2,029
requests — Tranco's head contains the ad-tech vendors' own sites) and
same-entity (1,920 — `ltwebstatic.com`/Shein, `muscache.com`/Airbnb,
`ebayimg.com`/eBay).

Party must be judged against the document that *loaded*, not the one
requested: Tranco lists `onelink.me`, which 301s to `www.appsflyer.com`, and
scoring against the pre-redirect URL made AppsFlyer's own first-party assets
look third-party — 14 MB of phantom blocking, and precision 0.805 instead of
0.950. `final_page_url()` follows the HAR redirect chain.

Two caveats. 599 of the 1,201 exceptions are `isPrivateBrowsingOnly`; the
crawl's blocking arm is a normal window with tracking protection forced on, so
it ignores them and blocks slightly more than a real Private Window would —
`FirefoxETPProxy(pbm=True)` models the latter. And the shipped `*-digest256`
lists are SHA-256 hashes that cannot be enumerated, so the proxy reads the
upstream shavar sources they are built from; the two can disagree at the
margins, which is consistent with the 52 remaining false negatives.

### Measured savings need a drift correction

Predicted savings count only the blocked request, while measured savings also
include the follow-up requests a blocked tracker never got to make, so
predicted should land *below* measured. It does not against the raw net
page-level delta of 37.5 MB — and that turns out to indict the measurement,
not the estimate. The 224 pages where ETP blocked nothing must show zero
saving in expectation, yet they total **−28.3 MB** of pure crawl-to-crawl
drift (ad rotation, changed content). The 273 pages it did act on total
65.7 MB; netting the drift out gives ~100 MB. So the headline 37.5 MB / 1.6%
figure *understates* the effect, and 51.8 MB predicted sits below both
drift-aware figures as it should.

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
