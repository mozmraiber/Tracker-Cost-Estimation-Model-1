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
};

/// What a tracker request would have cost, had it not been blocked:
/// `(bytes, cpu_ms)`.
///
/// [`estimator::estimate_resources`] verbatim; see it for what the arguments
/// mean, what the estimate is, and why none of them is optional. The one thing
/// this signature adds is that requiring all four is where it differs from
/// `xgb-classifier`'s, which defaults the last two, having grown them after
/// the fact.
///
/// A plain tuple rather than a struct with named fields: a `#[pyclass]` costs
/// ~18 KiB of type-object and getter machinery, which does not fit the size
/// budget `tests/test_browsing_journey.py` holds this extension to. The same
/// budget is why the table quantises its values into a codebook.
#[pyfunction]
fn estimate_resources(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
) -> PyResult<(i64, f64)> {
    Ok(estimator::estimate_resources(
        url,
        context.into(),
        initiator.into(),
        method,
    ))
}

#[pymodule]
fn llm_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let () = DISCRIMINANTS_MATCH;
    m.add_class::<RequestContext>()?;
    m.add_class::<RequestInitiator>()?;
    m.add_function(wrap_pyfunction!(estimate_resources, m)?)?;
    Ok(())
}
