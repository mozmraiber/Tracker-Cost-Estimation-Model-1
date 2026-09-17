//! Resource-cost estimates for blocked tracker requests, from the shipped
//! XGBoost artifact rather than from a lookup table.
//!
//! Near-identical interface to `llm-classifier` -- [`RequestContext`] and
//! [`estimate_resources`], returning `(bytes, cpu_ms)` -- and the same product
//! claim behind it: the dashboard reports a weekly *total*, so the answer has
//! to be a conditional mean whose sum is unbiased, not a median. Two optional
//! parameters are this crate's own; see [`estimate_resources`].
//!
//! What differs is where the answer comes from. `llm-classifier` compiles in a
//! hierarchy of conditional means over URL features and looks the request up.
//! This crate compiles in `models/per_request/xgb_transfer_bytes.json` -- the
//! `xgboost_shipped` predictor of `tests/solutions.py`, the one the paper
//! reports -- by transpiling its 370 trees to Rust with m2cgen and porting the
//! feature pipeline that feeds them. `scripts/build_model.py` does both.
//!
//! The point of having both is to price the model against the table on the one
//! axis a browser cares about and the paper's per-request metrics do not
//! measure: what it costs to ship. The answer is stark. Stripped, this
//! extension adds 13.8 MB to an empty PyO3 extension where `llm-classifier`
//! adds 49 KiB -- 277x, against the 50 KB budget
//! `tests/test_browsing_journey.py` holds that one to. Three quarters of the
//! excess is not even the trees: 10.0 MB is the truncated-SVD basis the 50
//! `url_emb_*` features are projected through, 50,000 vocabulary terms by 50
//! components, which cannot be quantised (see [`embedding`]) or pruned without
//! moving the predictions.
//!
//! Accuracy is the other half of the price, and it does not pay for the size
//! either. Over the two datasets `tests/conftest.py` runs, median journey
//! error is 8.2% and 30.3% here against the table's 5.1% and 5.7%, and the
//! second is biased low by its whole magnitude -- the defect the
//! mean-calibrated estimators exist to avoid. `tests/solutions.py` records it
//! in `KNOWN_UNCALIBRATED`.
//!
//! None of that is this crate's doing. Those are `xgboost_shipped`'s own
//! numbers, to two decimal places: the interface now carries every input the
//! artifact was fitted on, so the two are one predictor and the comparison is
//! the model against the table with nothing else in it. Getting there took
//! three things the interface did not originally have -- the request's
//! initiator, its HTTP method, and [`RequestContext::JSON`] -- which between
//! them were worth 23 points on the CSV extract.
//! `scripts/interface_ablation.py` prices each, and [`features::build`] says
//! what is left: a query string the journey harness cannot hand over, which
//! costs nothing measurable.
//!
//! What is *not* a source of error is the port. Given the inputs the interface
//! can carry, this crate's 80 features are bit-identical to
//! `engineer_features`' over every row of the CSV extract's test half, and
//! 99.8% of its estimates are within a byte of the artifact's own.
//! `tests/test_xgb_classifier_port.py` is that claim as a test. Getting there
//! took reproducing four things about scikit-learn's arithmetic exactly rather
//! than approximately, and correcting two that m2cgen gets wrong about this
//! model -- how it prints a float32 threshold, and where a log-link
//! objective's intercept belongs. [`embedding`] and `scripts/build_model.py`
//! record which.
//!
//! The CPU half of the answer is copied verbatim from `llm-classifier`,
//! coefficients and all. It is not fitted in either crate -- the training log
//! has no CPU column -- so reproducing it here rather than deriving something
//! new keeps the two crates differing in exactly one thing, the byte estimate.

use pyo3::prelude::*;

mod blob;
mod embedding;
mod features;
mod generated;
mod hash;
mod trees;
mod url;

/// Kind of subresource the request is fetching.
///
/// The variant selects the log `resource_type` the booster's `rt_*` one-hots
/// and the `domain_type_median` encoding are keyed on -- see
/// [`RequestContext::resource_type`].
///
/// `llm-classifier`'s first eleven variants and their discriminants, so a
/// caller of either can pass the same value, plus [`RequestContext::JSON`]
/// appended. The discriminants are what its generated table hashed into every
/// lookup key, which is why the shared ones keep the values they had and
/// anything new goes on the end.
#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RequestContext {
    SCRIPT = 0,
    IMAGE = 1,
    VIDEO = 2,
    OTHER = 3,
    AUDIO = 4,
    CSS = 5,
    FONT = 6,
    HTML = 7,
    TEXT = 8,
    WASM = 9,
    XML = 10,
    /// An XHR or `fetch` returning JSON, which the log records as its own
    /// `resource_type` and `llm-classifier` has no variant for -- its table was
    /// fitted with the two folded together, so it could not tell them apart if
    /// it were told. This crate can: 8% of the export's blocked requests are
    /// `json`, and reading the `other` row for them was the last of the
    /// interface's cost.
    JSON = 11,
}

impl RequestContext {
    /// The log's `resource_type` for this context.
    ///
    /// Total, and one-to-one with the values the log records: every
    /// `resource_type` in the training log has a variant, so no two of them
    /// share a row of the target encodings.
    fn resource_type(self) -> &'static str {
        match self {
            Self::SCRIPT => "script",
            Self::IMAGE => "image",
            Self::VIDEO => "video",
            Self::OTHER => "other",
            Self::AUDIO => "audio",
            Self::CSS => "css",
            Self::FONT => "font",
            Self::HTML => "html",
            Self::TEXT => "text",
            Self::WASM => "wasm",
            Self::XML => "xml",
            Self::JSON => "json",
        }
    }
}

/// What caused the browser to issue the request.
///
/// The artifact's `init_script`, `init_parser` and `init_other` one-hots, as
/// one value. Only those three appear in the feature matrix, so the log's
/// `preflight`, `FedCM` and `preload` -- 1.2% of the export between them --
/// have no variant and arrive as [`RequestInitiator::UNKNOWN`], which is what
/// `engineer_features` gives them too: all three one-hots zero.
///
/// This is the one thing worth the most to the estimator that
/// `llm-classifier`'s interface does not carry. Script-initiated tracker
/// requests average 12.2 KB and account for 87.7% of all blocked bytes;
/// parser-discovered ones average 2.2 KB. See the crate header.
#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum RequestInitiator {
    /// The HTML parser found the URL in the markup: `<img src>`, `<script
    /// src>`, `<link>`.
    PARSER = 0,
    /// JavaScript caused it: `fetch`, XHR, `new Image()`, an inserted tag.
    SCRIPT = 1,
    /// Recorded as `other` by the log.
    OTHER = 2,
    /// Not known, or none of the above. The default, and what the estimator
    /// answered from before the interface carried an initiator at all.
    #[default]
    UNKNOWN = 3,
}

/// What a tracker request would have cost, had it not been blocked:
/// `(bytes, cpu_ms)`.
///
/// `initiator` and `method` are optional, and their defaults reproduce exactly
/// what this function answered before they existed: an unknown initiator and
/// no method leave the four features derived from them zero, which is what
/// `engineer_features` produces for a log row that records neither. So a
/// two-argument call still works, and still matches `llm_classifier`'s
/// signature -- that crate's table is keyed on the URL and the context and has
/// no use for either, which is why it does not take them.
///
/// Supplying them is worth a lot: on the CSV extract it is the difference
/// between 28.7% and 9.1% median journey error. See the crate header.
///
/// A plain tuple rather than a struct with named fields, as `llm-classifier`
/// returns, so the two stay drop-in. Its reason for being a tuple was the
/// ~18 KiB a `#[pyclass]` costs; that argument does not survive an extension
/// this size, but matching the interface does.
#[pyfunction]
#[pyo3(signature = (url, context, initiator=RequestInitiator::UNKNOWN, method=None))]
fn estimate_resources(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: Option<&str>,
) -> PyResult<(i64, f64)> {
    let bytes = estimate(url, context, initiator, method);
    Ok((bytes, cpu_ms_for(bytes, context)))
}

/// The 80 features [`estimate_resources`] hands the booster, in the order the
/// artifact names them.
///
/// Not part of the estimator's interface -- `llm-classifier` has no
/// counterpart and callers have no use for it. It exists because this crate's
/// correctness claim is that its feature pipeline reproduces
/// `engineer_features`, and the only way to test that claim directly is to
/// compare the vectors rather than infer a disagreement from the predictions:
/// the ensemble is sensitive enough that a 1e-7 difference in one embedding
/// moves the answer, so a prediction mismatch says nothing about which of the
/// 80 features is wrong. `tests/test_xgb_classifier_port.py` asserts them
/// equal column by column.
#[pyfunction]
#[pyo3(signature = (url, context, initiator=RequestInitiator::UNKNOWN, method=None))]
fn feature_vector(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: Option<&str>,
) -> PyResult<Vec<f64>> {
    let parsed = url::Url::parse(url);
    Ok(features::build(&parsed, context.resource_type(), initiator, method).to_vec())
}

/// Main-thread milliseconds per KiB, indexed by [`RequestContext`]
/// discriminant.
///
/// PROVISIONAL, and copied from `llm-classifier` unchanged -- see the
/// `CPU_MS_PER_KIB` doc comment there for where the two anchors come from and
/// what would replace them. Neither crate fits this: the training log carries
/// `transfer_bytes` and no CPU column, so there is nothing per-URL to fit
/// against. Keeping the same coefficients is deliberate, so that a difference
/// between the two crates' CPU answers is a difference in their byte
/// estimates and nothing else.
static CPU_MS_PER_KIB: [f64; 12] = [
    2.0,  // SCRIPT  parse + compile + execute; the dominant tracker cost
    0.10, // IMAGE   decode, often but not always off the main thread
    0.02, // VIDEO   demux/decode is mostly hardware or off-thread
    0.10, // OTHER   catch-all for a type the log does not name
    0.02, // AUDIO   as VIDEO
    0.30, // CSS     parse plus the style recalc it forces
    0.05, // FONT    little beyond decode
    0.30, // HTML    parse and DOM construction
    0.20, // TEXT
    2.00, // WASM    compile-bound, as SCRIPT
    0.20, // XML
    0.10, // JSON    parse, then whatever the handler does with the object
];

/// Main-thread cost charged per request regardless of size.
///
/// A zero-byte beacon is not free: something still dispatched it and ran its
/// completion handler. 11% of the blocked requests in this repo's crawl
/// transferred no bytes at all, and without this term every one of them would
/// be estimated at zero CPU.
const CPU_MS_FIXED: f64 = 0.15;

/// CPU estimate for a request of `bytes` in `context`. See [`CPU_MS_PER_KIB`].
fn cpu_ms_for(bytes: i64, context: RequestContext) -> f64 {
    // Indexing is in range because the array has one slot per variant.
    let rate = CPU_MS_PER_KIB[context as usize];
    CPU_MS_FIXED + (bytes.max(0) as f64 / 1024.0) * rate
}

/// The estimator proper, kept free of Python types.
///
/// The three steps `xgboost_shipped` takes in Python, in order: build the
/// feature vector, run the booster, undo the training offset. The booster's
/// `reg:tweedie` objective has a log link, so its answer is
/// `base_score * exp(margin)` -- `base_score` is stored un-linked and m2cgen
/// does not apply the link, which is why the intercept is multiplied in here
/// rather than transpiled into the trees. `scripts/build_model.py --check`
/// verifies that against `model.predict`.
///
/// The `- 1` is the training script's offset: Tweedie needs a strictly
/// positive target, so the model was fitted on `transfer_bytes + 1`.
fn estimate(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: Option<&str>,
) -> i64 {
    let parsed = url::Url::parse(url);
    let features = features::build(&parsed, context.resource_type(), initiator, method);
    let bytes = generated::BASE_SCORE * trees::margin(&features).exp() - 1.0;
    // The clip is `np.clip(..., 0, None)`; the upper bound only guards against
    // a hand-edited payload, since the fitted leaves cannot reach it.
    bytes.round().clamp(0.0, i64::MAX as f64) as i64
}

#[pymodule]
fn xgb_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RequestContext>()?;
    m.add_class::<RequestInitiator>()?;
    m.add_function(wrap_pyfunction!(estimate_resources, m)?)?;
    m.add_function(wrap_pyfunction!(feature_vector, m)?)?;
    Ok(())
}
