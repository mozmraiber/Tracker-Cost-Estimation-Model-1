# xgb-classifier

The shipped XGBoost artifact behind `llm-classifier`'s interface, so the two
can be priced against each other.

`llm-classifier` is the estimator this project ships to the browser: a
hierarchy of conditional means over URL features, compiled in as a table. This
crate answers the same call with `models/per_request/xgb_transfer_bytes.json` —
the `xgboost_shipped` predictor of `tests/solutions.py`, the one the paper
reports — by transpiling its 370 trees to Rust with
[m2cgen](https://github.com/BayesWitnesses/m2cgen) and porting the
`engineer_features` pipeline that feeds them.

```python
estimate_resources(url, context, initiator=UNKNOWN, method=None) -> (bytes, cpu_ms)
```

`llm-classifier` takes the same four, and both enums are variant-for-variant
its own with the same discriminants, so one call site serves either and
`tests/solutions.py` measures them through identical inputs. It requires all
four where these default the last two, so a call written against it works here
and not the other way round. What is extra here exists because the model was
fitted on more than the table was: the request's initiator type and HTTP
method, and a `JSON` request context, which the log records as a resource type
of its own and the table cannot distinguish because it was fitted with `json`
and `other` binned together.

## The result

| | median journey error | | shipped size |
|---|---|---|---|
| | `csv` | `http_archive` | over an empty extension |
| `llm-classifier` | **5.1%** | **5.7%** | **49 KiB** |
| `xgb-classifier` | 8.2% | 30.3% | 13.8 MB |
| `xgboost_shipped` in Python | 8.2% | 30.3% | — |

The last two rows are the same numbers to two decimal places, and that is the
point: the interface now carries every input the artifact was fitted on, so the
crate *is* `xgboost_shipped` and the comparison is the model against the table
with nothing else in it. The table wins on both axes — 277x on size, 1.6x to
5.3x on error — and the 30.3% is biased low by its whole magnitude, which is
the defect the mean-calibrated estimators exist to avoid.

**The model is the expensive part, and it is the less accurate one.**
`tests/solutions.py` has recorded `xgboost_shipped` in `KNOWN_UNCALIBRATED`
since before this crate existed; compiling it in does not change that. What
compiling it in reveals is the size: 13.8 MB, of which only ~2.9 MB is the
trees. 10.0 MB is the truncated-SVD basis the 50 `url_emb_*` features are
projected through — 50,000 vocabulary terms by 50 components — and it cannot be
made smaller. Quantising it to 16 bits halves the payload and moves the median
prediction by 9.6%.

**What the interface had to carry to get there.**
`scripts/interface_ablation.py` degrades the Python predictor one input at a
time, then takes back each thing the interface gained, to price it:

```
$ python xgb-classifier/scripts/interface_ablation.py
csv: 13228 test rows, 7.8% json, 68.8% multi-parameter, 98.4% with an initiator type
journey seed 3, 600 journeys of 40 pages

  predictor                                      median    signed
  xgboost_shipped (the log's own columns)         7.64%    -6.30%
  query rebuilt out of filler = the crate         7.64%    -6.30%
  ...and without the initiator parameters        27.03%   -26.86%
  ...and without a JSON context                  28.71%   -28.65%

  xgb_classifier (the crate, measured)            7.64%    -6.30%
```

Three things closed a 21-point gap, and they were worth very different amounts:

* **The initiator and the method**, 19.4 points. `initiator_type` is the
  strongest non-URL signal the model has: script-initiated tracker requests
  average 12.2 KB and carry 87.7% of all blocked bytes, parser-discovered ones
  average 2.2 KB. They are now `estimate_resources` arguments.
* **A JSON request context**, 1.7 points. 8% of blocked requests are `json`,
  and they were reading the `other` row of both target encodings.
* **The rebuilt query**, nothing measurable. The log kept each URL's length and
  parameter count but not the query, and `tests/solutions.estimator_urls`
  rebuilds one matching both — spending the byte budget on `&` separators
  first, which makes `num_query_params` exact at identical length, so
  `llm-classifier`'s predictions do not move at all. What survives is
  `file_extension` on 1.6% of rows, which the SQL reads off the whole URL, so a
  real `?url=beacon.gif` sets it and filler cannot. That one is the harness's
  and would cost a browser nothing.

## It really is the same predictor

Given the inputs the interface can carry, the 80 features are **bit-identical**
to `engineer_features`' over every row of the CSV extract's test half, and
99.8% of the estimates are within a byte of the artifact's own. The remainder
differ by ~1e-5 relative — six bytes on a 600 KB asset — because m2cgen sums
the float32 leaves in float64 where XGBoost sums them in float32.
`tests/test_xgb_classifier_port.py` is that claim as a test, asserted column by
column.

Bit-exactness was not free, and the things that stood in the way are the
interesting part of the crate. The ensemble is pathologically sensitive: fitted
with `tree_method="hist"`, its split thresholds sit *on* observed feature
values rather than between them, so a single ulp of difference in an embedding
sends a request down a different branch. At 1 ulp out, a fifth of estimates
were wrong and the summed bytes moved 5%. Closing it took reproducing four
details of scikit-learn's arithmetic exactly — `f32` accumulation in scipy's
own visit order, the mixed `f32`/`f64` L2 normalisation, and the fused
multiply-add that scipy's C++ `axpy` compiles to — plus correcting two things
m2cgen gets wrong about this model:

* it prints float32 split thresholds with float32's shortest decimal and parses
  them back as `f64`, which is up to half a float32 ulp off the value the
  booster compares against. Combined with `hist`'s on-value thresholds, that
  alone moved 22% of estimates.
* it adds `base_score` straight into the boosting margin. XGBoost ≥1.7 stores
  `base_score` un-linked, so for this model's `reg:tweedie` the prediction is
  `base_score * exp(margin)`, not `exp(base_score + margin)`.

`src/embedding.rs` and `scripts/build_model.py` carry the details.

## Building

The generated files are not checked in — they are 24 MB derived from artifacts
that are themselves gitignored.

```sh
# writes src/trees.rs, src/model.bin and src/generated.rs
python xgb-classifier/scripts/build_model.py --check

cd xgb-classifier && maturin develop --release
```

`--check` verifies the transpiled trees against `model.predict` before writing
anything. The build needs `models/per_request/xgb_transfer_bytes.json` and
`url_embedder.joblib`, plus `data/raw/per_request_1pct.csv` for the two target
encodings; it says which is missing if one is.

Release builds use `opt-level = 1`. `src/trees.rs` is 13 MB of nested `if` in
370 functions, and the tree walk is branch-bound rather than something the
optimiser improves, so level 3 costs minutes for nothing.

## Layout

```
scripts/build_model.py        artifact -> src/trees.rs, src/model.bin, src/generated.rs
scripts/interface_ablation.py what estimate_resources' arguments cost the model
src/lib.rs                    the interface, the estimator, and the CPU estimate
src/features.rs               the 80-feature vector: a port of engineer_features
src/embedding.rs              the 50 url_emb_* features: TF-IDF + truncated SVD
src/url.rs                    the URL fields, as sql/05_per_request_full.sql derived them
src/blob.rs                   reader for the packed vocabulary, basis and encodings
src/hash.rs                   FNV-1a, the one hash both sides of the build must agree on
```

The CPU half of `(bytes, cpu_ms)` is copied verbatim from `llm-classifier`,
coefficients and all. Neither crate fits it — the training log has no CPU
column — so reproducing it keeps the two differing in exactly one thing.
