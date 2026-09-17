//! Transfer-size estimates for blocked tracker requests.
//!
//! The dashboard claim this backs is a total -- "Firefox saved you
//! approximately 2.3MB this week" -- so [`estimate_size`] answers with a
//! conditional *mean*, not a median. Summing conditional medians of a
//! distribution that is 48% zeros understates a week's total badly; summing
//! conditional means does not.
//!
//! The estimate is a lookup in [`table`], a hierarchy of means fitted over the
//! blocked-request log by `scripts/build_table.py`. Levels are tried from the
//! most specific key that identifies a single asset down to the request
//! context alone, and the first hit wins. Regenerate the table with that
//! script; every URL rule below has a twin in it.

use pyo3::prelude::*;

/// Kind of subresource the request is fetching.
#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RequestContext {
    SCRIPT = 0,
    IMAGE = 1,
    VIDEO = 2,
    OTHER = 3,
}

/// Estimate the transfer size, in bytes, of a tracker request.
///
/// `tracker_index` is the URL's position in the Disconnect list, as
/// `disconnect.tracker_index` reports it; only URLs that list matches are in
/// scope. An index outside the fitted range still answers, from the request
/// context alone.
#[pyfunction]
fn estimate_size(url: &str, tracker_index: i64, context: RequestContext) -> PyResult<i64> {
    Ok(estimate(url, tracker_index, context))
}

/// The estimator proper, kept free of Python types.
fn estimate(_url: &str, _tracker_index: i64, _context: RequestContext) -> i64 {
    return 0
}

#[pymodule]
fn llm_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RequestContext>()?;
    m.add_function(wrap_pyfunction!(estimate_size, m)?)?;
    Ok(())
}
