//! The 80-element feature vector the shipped booster expects.
//!
//! A port of `engineer_features` in `src/model/train_multi_target.py`, in the
//! order `xgb_transfer_bytes.json` names its features. The order is part of
//! the artifact -- `tests/solutions.py` asserts the Python feature frame still
//! matches it -- so the layout below is not free to change.
//!
//! Where the Python reads a column of the request log, this derives the same
//! quantity from the URL, reproducing the SQL in `sql/05_per_request_full.sql`
//! that produced the column. The two columns no URL can yield -- the request's
//! initiator type and HTTP method -- are parameters instead; [`build`] says
//! what happens when a caller does not have them.

use crate::blob;
use crate::embedding;
use crate::generated::{GLOBAL_MEDIAN, N_COMPONENTS};
use crate::hash::fnv1a64;
use crate::url::Url;
use crate::RequestInitiator;

/// Features the booster takes: 30 derived from the request, then the 50
/// `url_emb_*` components.
pub const N_FEATURES: usize = 30 + N_COMPONENTS;

/// `resource_type` values `engineer_features` one-hots, in its order.
const RESOURCE_TYPES: [&str; 6] = ["script", "image", "other", "html", "text", "css"];

/// `file_extension` values it one-hots, in its order.
const EXTENSIONS: [&str; 8] = ["js", "gif", "png", "jpg", "html", "php", "json", "css"];

/// `initiator_type` values it one-hots, in its order. The log's `preflight`,
/// `FedCM` and `preload` are none of these, and `engineer_features` leaves all
/// three zero for them -- which is what `RequestInitiator::UNKNOWN` does.
const INITIATORS: [RequestInitiator; 3] = [
    RequestInitiator::SCRIPT,
    RequestInitiator::PARSER,
    RequestInitiator::OTHER,
];

/// Build the feature vector for a request.
///
/// `resource_type` is the log value the caller's request context maps to; it
/// keys the `domain_type_median` encoding and the `rt_*` one-hots.
///
/// `initiator` and `method` are what the request's `initiator_type` and
/// `http_method` columns recorded; a caller that does not know them leaves
/// them at their defaults and gets the four features derived from them zero,
/// which is what `engineer_features` produces for a log row recording neither.
///
/// Two features can still come out differently from the log's, and both belong
/// to the journey harness rather than to this function. `has_query_params` and
/// the `ext_*` one-hots are read off the URL as given, which is right by
/// definition -- they are exactly what the SQL read off the recorded URL, so a
/// browser holding the real URL gets the real values. The harness does not hold
/// it: the log kept the query's length and parameter count but not the query,
/// so `tests/solutions.estimator_urls` rebuilds one matching both out of
/// filler, which has no dot-suffix for `file_extension` to find and loses its
/// `?` when the length budget rounds away. Measured, that is worth nothing at
/// all on journey totals.
///
/// `rt_other` and `domain_type_median` differ too, but for a reason that is
/// about the enum rather than this function -- see
/// `RequestContext::resource_type`.
pub fn build(
    url: &Url<'_>,
    resource_type: &str,
    initiator: RequestInitiator,
    method: Option<&str>,
) -> [f64; N_FEATURES] {
    let mut features = [0.0; N_FEATURES];

    // Target encodings, with `engineer_features`' fallback chain: the pair's
    // median, then the domain's, then the global one.
    let domain = blob::domain_median(fnv1a64(url.host.as_bytes()));
    let domain_type =
        blob::domain_type_median(fnv1a64(format!("{}\0{}", url.host, resource_type).as_bytes()));
    features[0] = domain.unwrap_or(GLOBAL_MEDIAN);
    features[1] = domain_type.or(domain).unwrap_or(GLOBAL_MEDIAN);

    features[2] = url.path_depth() as f64;
    features[3] = url.length() as f64;
    features[4] = url.num_query_params() as f64;
    features[5] = f64::from(u8::from(url.has_query()));

    for (i, kind) in RESOURCE_TYPES.iter().enumerate() {
        features[6 + i] = f64::from(u8::from(*kind == resource_type));
    }

    let extension = url.file_extension();
    for (i, ext) in EXTENSIONS.iter().enumerate() {
        features[12 + i] = f64::from(u8::from(extension.as_deref() == Some(*ext)));
    }

    for (i, kind) in INITIATORS.iter().enumerate() {
        features[20 + i] = f64::from(u8::from(*kind == initiator));
    }
    // `engineer_features` compares the log's `http_method` against 'POST'
    // exactly, and the log stores it upper-case; the comparison is loosened
    // here only so a caller passing 'post' is not silently wrong.
    features[23] = f64::from(u8::from(
        method.is_some_and(|m| m.eq_ignore_ascii_case("POST")),
    ));

    // The hand-crafted regex features, over `url_path` as the Python has them.
    // Every alternative of every pattern is listed, including the two the
    // pattern subsumes (`gtag` inside `tag`, `usersync` inside `sync`), so
    // each line reads against its regex one-for-one. Case-sensitive, as
    // `Series.str.contains` is by default.
    let path = url.path;
    let flag = |matched: bool| f64::from(u8::from(matched));
    features[24] = flag(contains_any(
        path, &[".js", "/js/", "script", "sdk", "lib", "tag", "gtm", "gtag"]));
    features[25] = flag(contains_any(path, &["collect", "beacon", "ping", "pixel", "track"]));
    features[26] = flag(contains_any(path, &[".gif", ".png", ".jpg", "pixel", "1x1"]));
    features[27] = flag(contains_any(path, &["sync", "match", "cookie", "usersync"]));
    features[28] = flag(contains_any(path, &["/ad/", "/ads/", "adserver", "pagead", "prebid"]));
    // `/v[0-9]/` is the one alternative that is not a literal.
    features[29] = flag(contains_any(path, &["/api/", "/collect", "/event"])
        || has_version_segment(path));

    features[30..].copy_from_slice(&embedding::embed(path));
    features
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

/// The `/v[0-9]/` alternative of the `path_has_api` pattern.
fn has_version_segment(path: &str) -> bool {
    path.as_bytes()
        .windows(4)
        .any(|w| w[0] == b'/' && w[1] == b'v' && w[2].is_ascii_digit() && w[3] == b'/')
}
