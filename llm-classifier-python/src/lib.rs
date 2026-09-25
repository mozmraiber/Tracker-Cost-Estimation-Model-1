//! The CPython extension over the `llm-classifier` estimator.
//!
//! All of it: the two argument enums, one function, and the module that
//! registers them. The estimator itself -- the table lookup, the URL features,
//! the CPU rates -- is the `llm-classifier` crate, which has no Python in it
//! and is what a browser would link. This crate is the only one that depends
//! on pyo3, and it exists so that split is a fact of the build rather than a
//! convention.
//!
//! It is also where the size budget lands: `tests/test_browsing_journey.py`
//! holds the built `.so` to 50 KB over an empty PyO3 extension, and what fills
//! it is the estimator's compiled-in table (35.8 KB of it) plus the pyo3
//! registration code for the two enums below. `Cargo.toml` and
//! `llm-classifier/scripts/build_table.py` record what each costs.
//!
//! The enums are declared twice, here and in the estimator, and that is the
//! one thing to be careful with in this crate. `#[pyclass]` has to sit on the
//! definition, so a pyo3-free estimator cannot also own the Python type. The
//! copies do not drift: [`From`] maps them variant by variant, so a variant
//! added on one side fails to compile, and the [`DISCRIMINANTS_MATCH`]
//! assertions pin the values, which is the part that matters -- they are
//! hashed into every lookup key, so a renumbering that type-checked would
//! silently return another asset's size.

use pyo3::prelude::*;

/// Kind of subresource the request is fetching.
///
/// Mirrors [`estimator::RequestContext`], which documents the variants and
/// owns the discriminants.
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
}

impl From<RequestContext> for estimator::RequestContext {
    fn from(context: RequestContext) -> Self {
        match context {
            RequestContext::SCRIPT => Self::SCRIPT,
            RequestContext::IMAGE => Self::IMAGE,
            RequestContext::VIDEO => Self::VIDEO,
            RequestContext::OTHER => Self::OTHER,
            RequestContext::AUDIO => Self::AUDIO,
            RequestContext::CSS => Self::CSS,
            RequestContext::FONT => Self::FONT,
            RequestContext::HTML => Self::HTML,
            RequestContext::TEXT => Self::TEXT,
            RequestContext::WASM => Self::WASM,
            RequestContext::XML => Self::XML,
        }
    }
}

/// What caused the browser to issue the request.
///
/// Mirrors [`estimator::RequestInitiator`], which documents the variants, owns
/// the discriminants, and says why the estimator takes one at all when it does
/// not key on it.
#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RequestInitiator {
    PARSER = 0,
    SCRIPT = 1,
    OTHER = 2,
    UNKNOWN = 3,
}

impl From<RequestInitiator> for estimator::RequestInitiator {
    fn from(initiator: RequestInitiator) -> Self {
        match initiator {
            RequestInitiator::PARSER => Self::PARSER,
            RequestInitiator::SCRIPT => Self::SCRIPT,
            RequestInitiator::OTHER => Self::OTHER,
            RequestInitiator::UNKNOWN => Self::UNKNOWN,
        }
    }
}

/// Whether this page load has already had a cascading tracker blocked on the
/// same host.
///
/// Mirrors [`estimator::CascadeRoot`], which documents the variants and says
/// why the third one is not a hedge. Unlike the two enums above this one has
/// no counterpart in `xgb-classifier`, which has no cascade model to key it
/// on; `estimate_resources` defaults it, so the two modules stay swappable.
#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CascadeRoot {
    FIRST = 0,
    REPEAT = 1,
    UNKNOWN = 2,
}

impl From<CascadeRoot> for estimator::CascadeRoot {
    fn from(root: CascadeRoot) -> Self {
        match root {
            CascadeRoot::FIRST => Self::FIRST,
            CascadeRoot::REPEAT => Self::REPEAT,
            CascadeRoot::UNKNOWN => Self::UNKNOWN,
        }
    }
}

/// Every variant of both enums holds the value the estimator gives it.
///
/// The `From` impls above already make a *missing* variant a compile error;
/// these make a renumbered one a compile error too, which is the failure that
/// would otherwise be silent. `scripts/build_table.py` hashes these same
/// numbers into the table's keys, so a mirror one off from the estimator would
/// answer every request with some other asset's size while type-checking
/// perfectly.
const DISCRIMINANTS_MATCH: () = {
    macro_rules! same {
        ($enum:ident: $($variant:ident),+ $(,)?) => {
            $(assert!($enum::$variant as u8 == estimator::$enum::$variant as u8);)+
        };
    }
    same!(RequestContext: SCRIPT, IMAGE, VIDEO, OTHER, AUDIO, CSS, FONT, HTML, TEXT, WASM, XML);
    same!(RequestInitiator: PARSER, SCRIPT, OTHER, UNKNOWN);
    same!(CascadeRoot: FIRST, REPEAT, UNKNOWN);
};

/// What a tracker request would have cost, had it not been blocked:
/// `(bytes, cpu_ms)`.
///
/// [`estimator::estimate_resources`] verbatim; see it for what the arguments
/// mean, what the estimate is, and why none of the first four is optional.
/// Requiring all four is where this signature differs from
/// `xgb-classifier`'s, which defaults the last two, having grown them after
/// the fact.
///
/// `include_followups` *is* defaulted, and to the answer this function gave
/// before it existed. It does not describe the request -- every caller knows
/// it, the way none of them reliably knows the initiator -- it chooses between
/// two costs, and a caller that has not thought about the difference wants the
/// direct one. Defaulting it is also what keeps this module interchangeable
/// with `xgb_classifier`, which has no cascade model to offer.
///
/// A plain tuple rather than a struct with named fields: a `#[pyclass]` costs
/// ~18 KiB of type-object and getter machinery, which does not fit the size
/// budget `tests/test_browsing_journey.py` holds this extension to. The same
/// budget is why the table quantises its values into a codebook.
#[pyfunction]
#[pyo3(signature = (url, context, initiator, method, include_followups = false,
                    cascade_root = CascadeRoot::UNKNOWN))]
fn estimate_resources(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
    include_followups: bool,
    cascade_root: CascadeRoot,
) -> PyResult<(i64, f64)> {
    Ok(estimator::estimate_resources(
        url,
        context.into(),
        initiator.into(),
        method,
        include_followups,
        cascade_root.into(),
    ))
}

/// How specifically the table recognised `url`, as a lower-case name.
///
/// [`estimator::classify`] verbatim. One of `path`, `template`, `prefix`,
/// `host`, `host_template`, `ext_query`, `ext`, `context` or `bodyless`, most
/// specific first; see [`estimator::UrlMatch`] for what each means and why
/// the estimator's own fallback rung is the classification worth exposing.
///
/// A `&'static str` rather than an enum, and that is the size budget talking
/// rather than taste: a third `#[pyclass]` would cost about 18 KiB of type
/// object and getters, which is a third of the whole extension's allowance.
/// The names are stable and `tests/top500.py` keys on them.
#[pyfunction]
#[pyo3(signature = (url, context, initiator, method))]
fn classify_url(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
) -> PyResult<&'static str> {
    Ok(estimator::classify(url, context.into(), initiator.into(), method).name())
}

#[pymodule]
fn llm_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let () = DISCRIMINANTS_MATCH;
    m.add_class::<RequestContext>()?;
    m.add_class::<RequestInitiator>()?;
    m.add_class::<CascadeRoot>()?;
    m.add_function(wrap_pyfunction!(estimate_resources, m)?)?;
    m.add_function(wrap_pyfunction!(classify_url, m)?)?;
    Ok(())
}
