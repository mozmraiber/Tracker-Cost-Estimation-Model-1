//! Resource-cost estimates for blocked tracker requests.
//!
//! The dashboard claim this backs is a total -- "Firefox saved you
//! approximately 2.3MB this week" -- so [`estimate_resources`] answers with a
//! conditional *mean*, not a median. Summing conditional medians of a
//! distribution that is 48% zeros understates a week's total badly; summing
//! conditional means does not.
//!
//! Two costs come back per request: transfer bytes, and main-thread CPU time.
//! They are not on the same footing. Bytes are a fitted lookup over a log of
//! real blocked requests; CPU is derived from the byte estimate by a
//! per-context rate, because no per-URL CPU labels are in the training log.
//! See [`cpu_ms_for`] for the coefficients and what would replace them.
//!
//! The estimate is a lookup in [`table`], a hierarchy of means fitted over the
//! blocked-request log by `scripts/build_table.py`. Levels are tried from the
//! most specific key that identifies a single asset down to the file extension
//! alone, and the first hit wins. Regenerate the table with that script; every
//! URL rule below has a twin in it.
//!
//! [`RequestContext`] enters twice, at two different resolutions. Lookup keys
//! carry it folded into the four groups of [`table::KEY_GROUP`], because the
//! path and extension already say most of what the exact context would, and
//! splitting entries 11 ways would strand the contexts the log barely covers.
//! A URL that matches no level at all is then answered by
//! [`table::CONTEXT_FALLBACK`], which *is* per-context: with no URL to go on,
//! the difference between a font and a video is all there is to go on.
//!
//! The estimator keys on the URL path and on the request host, but not on
//! which tracker served it. Asset paths are distinctive enough on their own --
//! `/en_US/fbevents.js` belongs to exactly one vendor -- that identifying the
//! *tracker* as well earns little for what it costs: it needs a table entry
//! per tracker, and it makes the caller resolve the Disconnect list first.
//! Measured on the journey suite, taking it out cost 0.8 points of median
//! error at an equal budget (7.1% -> 7.9%), of which dropping the
//! now-redundant per-tracker level won back 0.1 in freed space, for 7.8% net.
//!
//! The *host* is a different question, and the answer changed. It used to be
//! keyed on as a flat rung and measured worse than either (9.1%), because
//! 5,213 exact hosts fragmented the fixed budget that 513 Disconnect entries
//! did not. Two things fixed that. It sits below the path rungs rather than
//! competing with them, so it only answers what they miss; and it is keyed
//! through [`HostTemplate`], which collapses per-customer subdomains, so the
//! 5,213 hosts pool into far fewer keys and an unseen customer still resolves.
//! The gain ranking now spends 2,369 of 8,700 entries on the templated host
//! and 36 on exact hosts -- it keeps an exact host only where that host
//! differs from its own template.
//!
//! What this buys is out-of-sample accuracy, which the journey suite is the
//! wrong instrument for: it scores the log the table was fitted on, where
//! paths recur and the host is redundant, and there the change is flat
//! (median 5.73% -> 5.42% on the HTTP Archive half, 5.11% -> 5.18% on the
//! CSV). On a fresh crawl the paths are all new and the host is the only rung
//! that generalizes. `tests/test_top500_estimates.py` scores that case and
//! moved -6.2% -> -2.6% overall, and -18.7% -> -8.5% on the news and media
//! pages where those vendors concentrate.
//!
//! [`estimate_resources`] takes the four things `xgb-classifier`'s does: the
//! URL, the request context, the request's initiator and its HTTP method. The
//! last two are the ones no URL yields, and neither is worth much here -- for
//! the reason the tracker identity is not worth much either.
//!
//! The method is worth one thing. A HEAD or a CORS preflight cannot carry a
//! response body, so [`estimate`] answers those from the method and does not
//! consult the table at all: the table priced them at 1,262 bytes against the
//! 9 they average, reading the path each preflight shares with the POST it
//! precedes. That is 84x better in per-request error over the 1.37% of blocked
//! requests concerned, and it barely moves a weekly total, which those
//! requests are 0.0017% of -- 0.05 points of median journey error and 0.3 of
//! the CSV extract's p90. POST is not in that category: a beacon endpoint has
//! a path of its own, and the table reads it better than the method does.
//!
//! The initiator is worth nothing at this budget and is not keyed on. The path
//! already separates the 17.2 KB script-initiated GETs from the 3.5 KB
//! parser-discovered ones, predicting them at 17.3 KB and 3.7 KB without being
//! told which is which, so entries split by initiator displace more accuracy
//! than they correct. `scripts/build_table.py` records both measurements,
//! against `BODYLESS_METHODS` and `INITIATOR_UNUSED`.
//!
//! There is no Python in this crate. [`estimate_resources`] is a plain Rust
//! function over plain Rust types, which is the form the estimator would ship
//! in -- a browser calling it has no interpreter -- and it is also the half
//! that `scripts/build_table.py` has to reproduce rule for rule, which is
//! easier to read without an FFI layer through it. The CPython extension is
//! the `llm-classifier-python` crate next door: it depends on this one, mirrors
//! the two argument enums as `#[pyclass]`es, and is the only place pyo3
//! appears.

mod table;

/// Kind of subresource the request is fetching.
///
/// The discriminants are hashed into every lookup key, so renumbering these
/// invalidates the generated table -- which is why the original four keep the
/// values they had and the rest are appended. `scripts/build_table.py` mirrors
/// them, and so does `llm-classifier-python`, whose `#[pyclass]` copy is held
/// to them by assertion.
///
/// `OTHER` is the catch-all, and takes the resource types the log records but
/// this enum does not name (`json`, most notably); `scripts/build_table.py`
/// owns that mapping.
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

/// What caused the browser to issue the request.
///
/// Identical to `xgb-classifier`'s, discriminants included, so the two
/// extensions stay swappable. The log's `preflight`, `FedCM` and `preload` --
/// 1.2% of the export between them -- have no variant and belong in
/// [`RequestInitiator::UNKNOWN`].
///
/// Taken and not used. The table does not key on the initiator, because the
/// URL path carries what it would say; see the crate header, and
/// `INITIATOR_UNUSED` in `scripts/build_table.py` for the measurement that
/// settled it. It is in the interface because the two crates share one, and
/// because it is worth 19 points of journey error to `xgb-classifier`, whose
/// booster has no path-level table to carry the information instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum RequestInitiator {
    /// The HTML parser found the URL in the markup: `<img src>`, `<script
    /// src>`, `<link>`.
    PARSER = 0,
    /// JavaScript caused it: `fetch`, XHR, `new Image()`, an inserted tag.
    SCRIPT = 1,
    /// Recorded as `other` by the log.
    OTHER = 2,
    /// Not known, or none of the above.
    #[default]
    UNKNOWN = 3,
}

/// What a tracker request would have cost, had it not been blocked:
/// `(bytes, cpu_ms)`.
///
/// `bytes` is a fitted estimate; `cpu_ms` is derived from it. The table is
/// fitted only over URLs the Disconnect list matches, since those are the ones
/// ETP blocks, but the caller does not have to say which entry matched -- a URL
/// from outside that population still answers, from whatever level of the
/// hierarchy its path reaches.
///
/// Every argument is one a browser holds at the point it blocks a request:
/// `nsIChannel` and `nsILoadInfo` expose the URI, the content policy type, the
/// loading principal and the method, which is what the paper's deployment
/// section maps these onto. Nothing is optional, and `""` is how a caller that
/// genuinely does not know the method says so -- it reads the table exactly as
/// a GET does, which is what this function answered before it took a method at
/// all. Defaulting the method instead would quietly hand a CORS preflight the
/// estimate belonging to the POST it precedes.
pub fn estimate_resources(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
) -> (i64, f64) {
    let bytes = estimate(url, context, initiator, method);
    (bytes, cpu_ms_for(bytes, context))
}

/// Main-thread milliseconds per KiB, indexed by [`RequestContext`] discriminant.
///
/// PROVISIONAL. These are not fitted. The training log carries `transfer_bytes`
/// and no CPU column, so there is nothing per-URL to fit against; what is
/// modelled here is the shape everyone agrees on -- cost scales with bytes, and
/// the constant depends on what the bytes are -- with the constants set from
/// two anchors rather than from a regression:
///
///  * the widely-quoted ~1 ms/KiB for JavaScript parse and compile on mid-tier
///    mobile, before any execution;
///  * this repo's own paired crawl, where the CPU attributable to blocked
///    trackers was 84.1 s against 21.8 MB of directly-blocked bytes, i.e.
///    ~4.0 ms/KiB -- but that figure also carries the CPU of the subresources a
///    blocked tracker never got to request, so it is an upper bound on the cost
///    of the request itself. Spread over all the bytes the blocking saved
///    (65.7 MB at page level) the same CPU is ~1.3 ms/KiB.
///
/// The script rate below sits between those bounds. Summed over the 1,890
/// tracker requests that crawl's ETP arm blocked, these coefficients predict
/// 47.2 s against the 84.1 s measured, a blended 1.75 ms/KiB. Landing *under*
/// the measurement is the right direction and not a miscalibration: the
/// measured figure includes the subresources a blocked tracker never got to
/// request and execute, which a per-request estimate does not claim to cover.
///
/// Treat any single CPU estimate as order-of-magnitude, and prefer the byte
/// estimate when only one can be trusted.
///
/// To replace this with a fitted table: Lighthouse's `bootup-time` audit
/// reports `scripting`, `scriptParseCompile` and `total` per *script URL*, so
/// `sql/02_lighthouse_features.sql` already reaches the right data -- it just
/// groups it by host. A per-URL variant of that extract would give
/// `scripts/build_table.py` a second target to fit, and a second value array
/// alongside `table::VALUES` would carry it at one extra byte per entry.
/// Non-script contexts would still need a different source, since `bootup-time`
/// only covers scripts.
static CPU_MS_PER_KIB: [f64; 11] = [
    2.0,  // SCRIPT  parse + compile + execute; the dominant tracker cost
    0.10, // IMAGE   decode, often but not always off the main thread
    0.02, // VIDEO   demux/decode is mostly hardware or off-thread
    0.10, // OTHER   catch-all, incl. the `json` XHR payloads
    0.02, // AUDIO   as VIDEO
    0.30, // CSS     parse plus the style recalc it forces
    0.05, // FONT    little beyond decode
    0.30, // HTML    parse and DOM construction
    0.20, // TEXT
    2.00, // WASM    compile-bound, as SCRIPT
    0.20, // XML
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
/// `_initiator` is accepted and ignored; see [`RequestInitiator`].
fn estimate(url: &str, context: RequestContext, _initiator: RequestInitiator, method: &str) -> i64 {
    // A response to one of these carries no payload, so the URL has nothing to
    // say about its size and no level of the table is worth consulting. The
    // table would otherwise answer a preflight from the path it shares with
    // the POST it precedes, which is the error this exists to avoid. See
    // BODYLESS_METHODS in scripts/build_table.py.
    if is_bodyless(method) {
        return dequantize(table::BODYLESS_MEAN);
    }

    let parts = UrlParts::parse(url);
    let ctx = context as usize;
    // Keys carry the folded group, not the context itself; the fallback below
    // carries the context. Indexing is in range because both arrays have one
    // slot per variant.
    let group = u32::from(table::KEY_GROUP[ctx]);

    let template = PathTemplate::new(parts.path);
    let ext = parts.extension();

    for level in Level::ALL {
        let key = key_of(level, group, &parts, &template, &ext);
        if let Some(value) = table_lookup(key) {
            return dequantize(value);
        }
    }
    // Nothing in the URL matched, so the context is all that is left to answer
    // from -- and it answers on its own terms: a font and a video have little
    // in common, and one mean over both is a worse guess than either.
    // Indexing is in range because the array has one slot per variant.
    dequantize(table::CONTEXT_FALLBACK[ctx])
}

// --------------------------------------------------------------------------- //
// Lookup keys
// --------------------------------------------------------------------------- //

/// One rung of the fallback hierarchy, most specific first.
///
/// The discriminants are part of every key, so reordering or renumbering these
/// invalidates the generated table.
#[derive(Clone, Copy)]
enum Level {
    /// The exact path: one specific asset, e.g. `/en_US/fbevents.js`.
    Path = 0,
    /// The path with cache-busting versions and hashes collapsed, so the entry
    /// survives the vendor rolling a new build.
    Template = 1,
    /// The first two templated segments plus the extension: the asset family.
    Prefix = 2,
    /// The request host alone. Sits below the path rungs and above the
    /// extension ones because that is what it is worth: it cannot pin an
    /// individual asset, but "a script from cdn.taboola.com" (158 KB mean) is
    /// a far better guess than "a .js from anywhere" (37 KB).
    ///
    /// This is the rung that carries an unseen path on a known host, which is
    /// the normal case on a live page -- ad and content-recommendation vendors
    /// serve versioned bundles, so the exact path in the training log is gone
    /// by the next build. Keying on the *Disconnect entry* was measured and
    /// rejected (see the module docs) because it made the caller resolve the
    /// tracker first; the host costs nothing extra, since the URL is already
    /// in hand.
    Host = 3,
    /// The host with its generated labels collapsed, so a per-customer
    /// subdomain still resolves. `#.edge.permutive.app` pools twenty-odd UUID
    /// hosts that average 179 KB; keyed exactly, every UUID the log never saw
    /// falls through to the extension rungs and is answered at 31 KB.
    ///
    /// The path rungs' [`PathTemplate`] relationship, one level up the host
    /// axis: `Host` is to `HostTemplate` as `Path` is to `Template`, and it
    /// reuses the same "this label was generated" rule.
    HostTemplate = 4,
    /// Extension and query length, which is the only handle on the media CDNs'
    /// per-request image and video URLs.
    ExtQuery = 5,
    /// Extension alone. The coarsest rung that is a table entry: a URL that
    /// misses even this is answered by `table::CONTEXT_FALLBACK`, which is the
    /// per-context mean and needs no lookup.
    Ext = 6,
}

impl Level {
    const ALL: [Level; 7] = [
        Level::Path,
        Level::Template,
        Level::Prefix,
        Level::Host,
        Level::HostTemplate,
        Level::ExtQuery,
        Level::Ext,
    ];
}

/// Hash the key for one level. The bytes fed to the hasher are exactly the
/// strings `key_strings` builds in `scripts/build_table.py`.
///
/// `group` is the folded request-context group from [`table::KEY_GROUP`], not a
/// [`RequestContext`] discriminant -- several contexts share one.
fn key_of(
    level: Level,
    group: u32,
    parts: &UrlParts<'_>,
    template: &PathTemplate<'_>,
    ext: &Extension,
) -> u32 {
    let mut h = Fnv1a::new();
    h.write_dec(level as u32);
    h.write_sep_dec(group);
    match level {
        Level::Path => h.write_sep_bytes(parts.path.as_bytes()),
        Level::Template => template.hash_into(&mut h, PathTemplate::ALL_SEGMENTS),
        Level::Prefix => {
            h.write_sep_bytes(ext.as_bytes());
            template.hash_into(&mut h, 2);
        }
        Level::Host => h.write_sep_bytes_lower(parts.host.as_bytes()),
        Level::HostTemplate => HostTemplate::new(parts.host).hash_into(&mut h),
        Level::ExtQuery => {
            h.write_sep_bytes(ext.as_bytes());
            h.write_sep_dec(query_bucket(parts.query_len));
        }
        Level::Ext => h.write_sep_bytes(ext.as_bytes()),
    }
    h.fold()
}

/// log2 bucket of the query-string length, including the `?`.
///
/// A coarse stand-in for the transform parameters the media CDNs carry
/// (`stp=dst-jpg_e35_s640x640`), which the log does not record verbatim.
fn query_bucket(query_len: usize) -> u32 {
    const MAX: u32 = 12;
    let floor_log2 = usize::BITS - 1 - (query_len + 1).leading_zeros();
    floor_log2.min(MAX)
}

/// Look a key up in the bucketed table.
///
/// The table is split into 256 buckets by the key's top byte, so an entry only
/// has to store the remaining 24 bits -- as a `KEY_HI` byte and a `KEY_LO`
/// halfword, kept in parallel arrays so neither needs an unaligned load. With
/// the value a byte of codebook index, that is 4 bytes an entry instead of the
/// 6 a flat `[u32]`/`[u16]` pair costs, which is what pays for the extra
/// entries inside the size budget. No key bits are dropped: bucket plus
/// remainder is the whole 32-bit hash, so a miss is still a miss.
fn table_lookup(key: u32) -> Option<u16> {
    let bucket = (key >> 24) as usize;
    let start = table::BUCKETS[bucket] as usize;
    let end = table::BUCKETS[bucket + 1] as usize;

    let (hi, lo) = ((key >> 16) as u8, key as u16);

    // Partition point of the bucket's (KEY_HI, KEY_LO) pairs, which the
    // generator emits in ascending key order.
    let (mut low, mut high) = (start, end);
    while low < high {
        let mid = low + (high - low) / 2;
        if (table::KEY_HI[mid], table::KEY_LO[mid]) < (hi, lo) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    match low < end && table::KEY_HI[low] == hi && table::KEY_LO[low] == lo {
        true => Some(table::CODEBOOK[table::VALUES[low] as usize]),
        false => None,
    }
}

/// Whether a response to `method` carries no body.
///
/// Compared case-insensitively, though every method a browser reports is the
/// upper-case form the HTTP method registry defines; the comparison is
/// loosened only so a caller passing `head` is not silently charged for a
/// payload. These two methods are all this needs to cover:
/// `scripts/build_table.py` fits `BODYLESS_MEAN` over exactly them, and POST
/// is deliberately not one of them.
fn is_bodyless(method: &str) -> bool {
    method.eq_ignore_ascii_case("HEAD") || method.eq_ignore_ascii_case("OPTIONS")
}

/// Undo the table's `log1p(bytes) * VALUE_SCALE` fixed-point encoding.
fn dequantize(value: u16) -> i64 {
    let bytes = (f64::from(value) / table::VALUE_SCALE).exp_m1();
    // The largest stored value is well under i64::MAX, so this only guards
    // against a hand-edited table.
    bytes.round().max(0.0).min(i64::MAX as f64) as i64
}

// --------------------------------------------------------------------------- //
// URL features
// --------------------------------------------------------------------------- //

/// The pieces of a request URL the estimator keys on.
struct UrlParts<'a> {
    /// Authority as written, without the scheme. Lowercased when hashed rather
    /// than here, so parsing stays allocation-free.
    host: &'a str,
    path: &'a str,
    /// Length of the query, counting the `?`; 0 when there is none. The log
    /// records only the length, which is why this is not the query itself.
    query_len: usize,
}

impl<'a> UrlParts<'a> {
    /// Split a URL without validating it: a bare `host/path` works, and a URL
    /// with no path at all is treated as `/`, matching the log's encoding.
    fn parse(url: &'a str) -> Self {
        let after_scheme = match url.find("://") {
            Some(i) => &url[i + 3..],
            None => url,
        };
        let start = after_scheme
            .find(['/', '?', '#'])
            .unwrap_or(after_scheme.len());
        let authority = &after_scheme[..start];
        let authority_stripped = &after_scheme[start..];
        // Userinfo and port are not part of what the log grouped on.
        let host = match authority.rfind('@') {
            Some(i) => &authority[i + 1..],
            None => authority,
        };
        let host = match host.rfind(':') {
            Some(i) if !host[i + 1..].contains(']') => &host[..i],
            _ => host,
        };

        let path_end = authority_stripped
            .find(['?', '#'])
            .unwrap_or(authority_stripped.len());
        let path = &authority_stripped[..path_end];

        Self {
            host,
            path: if path.is_empty() { "/" } else { path },
            // A fragment is not part of what the log measured, but a `#` in a
            // subresource URL is rare enough that the tail is taken whole.
            query_len: match authority_stripped[path_end..].starts_with('?') {
                true => authority_stripped.len() - path_end,
                false => 0,
            },
        }
    }

    /// Text after the last dot of the last path segment, lowercased.
    ///
    /// Empty when the last segment has no dot, or when the tail is longer than
    /// an extension holds -- past [`Extension::MAX`] it is a content hash or a
    /// stray parameter, not a file type.
    fn extension(&self) -> Extension {
        let segment = match self.path.rfind('/') {
            Some(i) => &self.path[i + 1..],
            None => self.path,
        };
        match segment.rfind('.') {
            Some(i) => Extension::new(&segment.as_bytes()[i + 1..]),
            None => Extension::EMPTY,
        }
    }
}

/// A lowercased file extension, inline: at this length it is never worth an
/// allocation, and `estimate_resources` is called once per blocked request.
struct Extension {
    bytes: [u8; Self::MAX],
    len: usize,
}

impl Extension {
    const MAX: usize = 8;
    const EMPTY: Self = Self {
        bytes: [0; Self::MAX],
        len: 0,
    };

    fn new(tail: &[u8]) -> Self {
        if tail.is_empty() || tail.len() > Self::MAX {
            return Self::EMPTY;
        }
        let mut ext = Self::EMPTY;
        for (slot, &b) in ext.bytes.iter_mut().zip(tail) {
            *slot = b.to_ascii_lowercase();
        }
        ext.len = tail.len();
        ext
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// A path with its cache-busting parts collapsed to `#`.
///
/// Tracker assets are versioned: `/pagead/managed/js/adsense/m202608060101/
/// show_ads_impl_fy2021.js` and `/modules.75bf3488a101739acfe9.js` are the same
/// asset every week under a new name. Templating them lets one table entry keep
/// matching after the vendor rolls a new build, instead of costing an entry per
/// build and missing next week's.
///
/// Nothing is allocated: the segments are hashed in place, so the template only
/// ever exists as the bytes fed to the hasher.
struct PathTemplate<'a> {
    path: &'a str,
}

impl<'a> PathTemplate<'a> {
    /// `segments` value meaning "the whole path".
    const ALL_SEGMENTS: usize = usize::MAX;

    fn new(path: &'a str) -> Self {
        Self { path }
    }

    /// Hash the first `segments` path segments of the template, separated by
    /// `/` as the generator writes them. The leading `/` of an absolute path
    /// counts as the first, empty, segment.
    fn hash_into(&self, h: &mut Fnv1a, segments: usize) {
        h.write_sep();
        for (i, segment) in self.path.split('/').enumerate() {
            if i > segments {
                break;
            }
            if i > 0 {
                h.write(b"/");
            }
            Self::hash_segment(h, segment);
        }
    }

    fn hash_segment(h: &mut Fnv1a, segment: &str) {
        Self::hash_segment_cased(h, segment, false);
    }

    /// `lower` ASCII-lowercases what is written, for the host rungs: the
    /// generator lowercases hosts in SQL and paths not at all.
    fn hash_segment_cased(h: &mut Fnv1a, segment: &str, lower: bool) {
        if is_hexish(segment) {
            h.write(b"#");
            return;
        }
        // Split into maximal alphanumeric tokens; a token that looks generated
        // becomes `#` and everything else is kept verbatim, punctuation
        // included.
        let bytes = segment.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if !bytes[i].is_ascii_alphanumeric() {
                h.write_cased(&bytes[i..i + 1], lower);
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let token = &bytes[start..i];
            if is_generated_token(token) {
                h.write(b"#");
            } else {
                h.write_cased(token, lower);
            }
        }
    }
}

/// A hostname with its generated labels collapsed, the host-axis twin of
/// [`PathTemplate`]. Mirrors `normalize_host` in scripts/build_table.py.
struct HostTemplate<'a> {
    host: &'a str,
}

impl<'a> HostTemplate<'a> {
    fn new(host: &'a str) -> Self {
        Self { host }
    }

    /// Hash the templated host, labels separated by `.` as the generator
    /// writes them, lowercased to match its `lower()`.
    fn hash_into(&self, h: &mut Fnv1a) {
        h.write_sep();
        for (i, label) in self.host.split('.').enumerate() {
            if i > 0 {
                h.write(b".");
            }
            PathTemplate::hash_segment_cased(h, label, true);
        }
    }
}

/// Whether a whole segment is a hex string or UUID, e.g.
/// `a8ff32f4-78c7-4428-825d-0badb488b68b`.
fn is_hexish(segment: &str) -> bool {
    segment.len() >= 8
        && segment.bytes().any(|b| b.is_ascii_digit())
        && segment.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')
}

/// Whether an alphanumeric token is a version or content hash: a long run of
/// digits, or a long mixed run that carries at least one digit. `en_US` and
/// `fy2021` are kept; `m202608060101` and `75bf3488a101739acfe9` are not.
fn is_generated_token(token: &[u8]) -> bool {
    let digits = token.iter().filter(|b| b.is_ascii_digit()).count();
    if digits == token.len() {
        token.len() >= 4
    } else {
        token.len() >= 8 && digits > 0
    }
}

// --------------------------------------------------------------------------- //
// Hashing
// --------------------------------------------------------------------------- //

/// FNV-1a, chosen because the generator has to reproduce it exactly in Python.
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    /// Fold to 32 bits, which is the key width the table stores.
    fn fold(&self) -> u32 {
        ((self.0 >> 32) ^ (self.0 & 0xffff_ffff)) as u32
    }

    fn write_sep(&mut self) {
        self.write(b"|");
    }

    fn write_sep_bytes(&mut self, bytes: &[u8]) {
        self.write_sep();
        self.write(bytes);
    }

    /// ASCII-lowercase each byte as it is fed in. Hosts are case-insensitive
    /// and the generator lowercases them in SQL, so the two must agree.
    fn write_sep_bytes_lower(&mut self, bytes: &[u8]) {
        self.write_sep();
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b.to_ascii_lowercase())).wrapping_mul(Self::PRIME);
        }
    }

    fn write_sep_dec(&mut self, value: u32) {
        self.write_sep();
        self.write_dec(value);
    }

    /// `write`, optionally ASCII-lowercasing each byte.
    fn write_cased(&mut self, bytes: &[u8], lower: bool) {
        match lower {
            true => {
                for &b in bytes {
                    self.0 =
                        (self.0 ^ u64::from(b.to_ascii_lowercase())).wrapping_mul(Self::PRIME);
                }
            }
            false => self.write(bytes),
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Self::PRIME);
        }
    }

    /// Write `value` as decimal digits, as Python's `str(int)` would.
    fn write_dec(&mut self, value: u32) {
        let mut buf = [0u8; 10];
        let mut n = 0;
        let mut rest = value;
        loop {
            buf[n] = b'0' + (rest % 10) as u8;
            rest /= 10;
            n += 1;
            if rest == 0 {
                break;
            }
        }
        buf[..n].reverse();
        self.write(&buf[..n]);
    }
}
