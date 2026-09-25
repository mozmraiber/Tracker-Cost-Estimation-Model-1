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
//! Both can be answered for the blocked request alone, or for the blocked
//! request *and the requests it would have gone on to make*, which is what
//! [`estimate_resources`]' `include_followups` selects. The second is what a
//! page measurably sheds when ETP blocks a loader script, and the first is
//! what a per-request observation can be compared against, so neither is the
//! right answer to both questions. The direct estimate is the fitted one; the
//! cascade on top of it is a calibrated constant with a wide error bar, and
//! [`FOLLOWUP_BYTES_PER_REQUEST`] is explicit about how wide.
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
//! moved -7.1% -> -2.5% overall, and -18.7% -> -6.5% on the news and media
//! pages where those vendors concentrate.
//!
//! [`estimate_resources`] takes the four things `xgb-classifier`'s does: the
//! URL, the request context, the request's initiator and its HTTP method. The
//! last two are the ones no URL yields, and neither is worth much here -- for
//! the reason the tracker identity is not worth much either. It takes one more
//! that `xgb-classifier` does not, `include_followups`, which is not a feature
//! at all: it selects which cost is being asked for, not what is known about
//! the request. The extension defaults it to false, so the shared interface is
//! unchanged for every caller that does not ask.
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

/// Whether this page load has already had a cascading tracker blocked on the
/// same host.
///
/// The one thing `estimate_resources` takes that is about the page rather than
/// the request, and the only one it cannot default away without cost. Two
/// blocked requests to `doubleclick.net` on one page do not prune two
/// disjoint subtrees -- they prune one, twice -- so charging both the full
/// cascade double-counts it. The crawl says that is worth about a quarter of
/// the cascade: 7,089 cascading blocks fall on 5,441 distinct host-page-load
/// triples, so 23% of them are repeats.
///
/// The variants are what a caller can honestly say, not a tri-state for
/// tidiness, and the third is the important one. Firefox knows this at block
/// time -- it already counts blocked trackers per top-level document for the
/// shield UI -- but an offline caller scoring a request log row by row does
/// not, and making it guess would be worse than letting it say so. So
/// [`CascadeRoot::UNKNOWN`] charges [`FOLLOWUP_BYTES_PER_REQUEST`], the
/// crawl-average over both cases, which is exactly what this function
/// answered before the variant existed.
///
/// See [`FOLLOWUP_BYTES_PER_HOST`] for the measurement and for why it is the
/// best-evidenced refinement to the cascade this repo has found.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CascadeRoot {
    /// No cascading tracker request to this host has been blocked on this
    /// page load yet, so this one prunes a subtree of its own.
    FIRST = 0,
    /// One has, so this request is inside a subtree already charged for.
    REPEAT = 1,
    /// The caller does not track blocked hosts per page and is saying so.
    #[default]
    UNKNOWN = 2,
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
/// The first four arguments are all things a browser holds at the point it
/// blocks a request: `nsIChannel` and `nsILoadInfo` expose the URI, the content
/// policy type, the loading principal and the method, which is what the paper's
/// deployment section maps these onto. None of them is optional, and `""` is
/// how a caller that genuinely does not know the method says so -- it reads the
/// table exactly as a GET does, which is what this function answered before it
/// took a method at all. Defaulting the method instead would quietly hand a
/// CORS preflight the estimate belonging to the POST it precedes.
///
/// `include_followups` is the one argument that is not about *this* request.
/// With it false the answer is the request's own transfer and its own CPU, and
/// nothing else -- the historical behaviour, and the right answer to "how big
/// was the thing ETP refused". With it true the answer also carries the
/// requests that request would have gone on to make and never did: a blocked
/// tag manager is not merely 93 KB of script, it is also the ad and analytics
/// stack it would have pulled in behind itself. See
/// [`FOLLOWUP_BYTES_PER_REQUEST`] for what that costs and how well it is known,
/// which is: an order of magnitude better than not modelling it at all, and a
/// good deal worse than the direct estimate beside it.
///
/// Which one a caller wants follows from what it is summing against. A
/// dashboard saying "Firefox saved you 2.3 MB this week" is claiming bytes that
/// did not cross the network, and the follow-ups did not cross it either, so
/// that claim wants `true`. A per-request comparison against an observed
/// response size wants `false`, because the observation covers one request.
/// `tests/test_top500_estimates.py` uses both, one per ground truth it has.
///
/// `cascade_root` is read only when `include_followups` is true, and says
/// whether this page load has already had a cascading tracker blocked on the
/// same host -- the difference between pruning a subtree and pruning a branch
/// of one already counted. [`CascadeRoot::UNKNOWN`] is the honest answer for a
/// caller that does not track that, and reproduces exactly what this function
/// returned before the argument existed. See [`CascadeRoot`] and
/// [`FOLLOWUP_BYTES_PER_HOST`].
pub fn estimate_resources(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
    include_followups: bool,
    cascade_root: CascadeRoot,
) -> (i64, f64) {
    let (bytes, matched) = resolve(url, context, initiator, method);
    let cpu_ms = cpu_ms_for(bytes, context);
    if !include_followups {
        return (bytes, cpu_ms);
    }
    // A method that forbids a response body prunes nothing: there is no code
    // to run and no document to parse, whatever the URL looks like. This
    // mattered less when the cascade scaled with the estimate -- the 9 bytes
    // such a request is priced at bought 21 bytes of subtree -- and matters
    // now that it does not, since a CORS preflight would otherwise be charged
    // the 47 KB an average blocked script prunes.
    let followup_bytes = if is_bodyless(method) {
        0
    } else {
        followup_bytes_for(bytes, context, matched, cascade_root,
                           UrlParts::parse(url).host)
    };
    (
        bytes + followup_bytes,
        cpu_ms + followup_cpu_ms_for(followup_bytes),
    )
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

/// Transfer bytes of the requests a blocked request would itself have made,
/// per blocked request, indexed by [`RequestContext`] discriminant.
///
/// This is the cascade the paper's limitations section names and declines to
/// quantify: block a tag manager, a consent provider or an ad loader and the
/// third parties it would have fetched are never fetched either, so the bytes
/// that did not cross the network are strictly more than the ones ETP refused.
///
/// PER REQUEST, NOT PER BYTE. Until 2026-09 this array held a multiple of the
/// blocked request's own size -- 2.3 bytes of subtree per byte refused -- on
/// the reasoning that a bigger script does more. Fit both terms against the
/// paired crawl at once and the size term all but vanishes: against the quiet
/// Disconnect-only delta, 27.9 KB per cascading request (95% [2.1, 48.3])
/// against 0.32 per cascading byte (95% [0.00, 1.14]), and with each page's
/// category demeaned out, 21.4 KB against 0.27. Cut the blocked requests by
/// their own size instead and the implied per-byte factor falls monotonically
/// -- 3.79 for requests under 10 KB, 2.63 from 10 to 50 KB, 0.37 above 50 KB.
/// A proportional model has to read that as a big script cascading less per
/// byte; a per-request model reads it as the subtree not caring much how big
/// its root was, which is what a tag manager looks like: the 2 KB `gtag/js`
/// stub and the 106 KB `fbevents.js` bundle both go on to fetch a stack.
///
/// Graded head to head with the crawl's cascade total held fixed -- so
/// neither form can win by being bigger, only by putting the same bytes in
/// better places -- per request beats per byte against the quiet delta, 82%
/// of paired bootstraps on per-page error and 59% on category totals, the
/// category error falling from 90.4 MB to 70.8.
///
/// BE PLAIN ABOUT HOW STRONG THAT IS NOW, because it is weaker than this
/// comment used to say and the numbers above are the pooled crawl's rather
/// than the single pass's. Three of the four figures moved against the form:
/// the per-byte term is no longer pinned to zero, the chance the size
/// ordering is the other way round went from 1% to 22%, and the loud
/// whole-page delta has *flipped* -- it now prefers per byte on per-page
/// error, 96% of bootstraps, where it once backed per request at 93%. So the
/// case for the shipped form rests on the quiet instrument alone, at 82%
/// against the 83% this file elsewhere treats as noise, with the loud one
/// dissenting. It is kept because the quiet instrument is the one with 5.1%
/// standard error against 23.4%, and because the dissent survives the
/// stability filter that repairs the loud instrument everywhere else in this
/// comment -- which makes it a real disagreement about shape rather than the
/// six pages again. A crawl an order of magnitude larger is what would settle
/// it.
///
/// Two refinements between the two forms were measured on the pooled crawl
/// and neither ships; `scripts/fit_followups.py` re-measures both every run.
///
///   Scaling with the blocked request's own size, at an exponent between the
///   0 the array ships and the 1 the old form used. The crawl does have a
///   signal there -- per request, the quiet instrument reads 28.0, 24.4,
///   54.6 and 51.4 KB over blocked requests under 5 KB, 5-20, 20-60 and
///   60 KB and up, so a large blocked script cascades about twice what a
///   small one does. Twice is not twenty times, which is why proportional
///   loses; it is also under this crawl's resolution. Levelled and graded,
///   the quiet instrument prefers the flat form at every exponent from 0.10
///   up (15% to 27% of bootstraps on per-page error) and the loud one likes
///   it at 99%, which is the same dissent as above.
///
///   Saturation: two blocked loaders on a page do not prune two disjoint
///   subtrees, since they share an ad stack, so a count should over-charge a
///   heavily blocked page. The bucket table leans that way and much harder
///   than the single pass could show -- 108 KB a request on the 55 pages
///   with one cascading block per pass, against 24 to 69 KB on the rest --
///   but a page-level intercept divided by k falls in k too, so the shape
///   has to be graded where it would be applied. Charge `k ** gamma`,
///   levelled: the quiet instrument prefers the flat count at every exponent
///   and only nears a tie at gamma 0.95, by nearly being it. This used to be
///   recorded here as a loose end left open; it is now rejected, and the
///   grading rather than the bucket table is why.
///
/// Only code can cascade, so only the two contexts that run code carry a
/// figure. HTTP Archive says the same thing outright: 88.4% of the bytes of
/// script-initiated tracker requests -- the descendants -- are themselves
/// scripts and 6.0% are documents, while images, fonts and video are
/// overwhelmingly parser-initiated, which is to say they are leaves the
/// markup asked for and that ask for nothing in turn. An image, a font, a
/// beacon or a media segment is left at zero rather than given a small figure
/// for tidiness: a pixel really does cost only its own bytes. CSS is the one
/// arguable zero (it can pull fonts and background images), and it stays zero
/// because no tracking block in the crawl below was a stylesheet.
///
/// CALIBRATED, NOT FITTED. `scripts/fit_followups.py` reproduces all of the
/// below, and the short version is that one dataset can measure the quantity
/// and cannot pin it down, while the other can pin down a bound and not the
/// quantity.
///
/// The paired top-500 crawl (`tests/top500.py`) has the counterfactual: one
/// arm blocked, the other did not, and the population is exactly what this
/// array is applied to. It is now ten pooled passes of that crawl rather than
/// one, and the difference decided the figure below, so it is worth being
/// exact about what moved.
///
/// The crawl offers the quantity two ways. Its *whole-page* byte delta --
/// everything the blocking arm did not fetch, wherever it came from -- is the
/// one this array was set from: on a single pass it read 70.2 KB a request
/// drift-corrected, which is where 70 KB came from. Its *listed-only* delta
/// -- the same difference counted over requests the Disconnect list names --
/// reads 43.7 KB raw, 47.0 KB drift-corrected, and 49.5 KB from the request
/// sets rather than the byte totals.
///
/// Pooling ten passes put the whole-page figure at 26.1 KB, 95% [-26, 70],
/// which cannot be read as a measurement of anything: it counts a superset of
/// what the listed-only delta counts and comes out below it. That is what
/// dropped the shipped figure from 70 to 47 -- believe the quiet instrument,
/// declare the loud one too noisy to speak -- and it was half right.
///
/// The loud instrument was not noisy. It was six pages. `page_delta_instability`
/// in `tests/top500.py` scores each page on how far its delta moves from pass
/// to pass relative to its own size, which is a thing only a repeated crawl
/// can compute, and six of 271 score above 1.0 -- their two arms differ, load
/// to load, by about as much as the page weighs. Over the other 265 the
/// whole-page delta reads 47.6 KB a request, 95% [12, 85], and the ordering
/// holds. The listed-only figure under the same filter moves 47.0 to 46.1,
/// which is the control: a filter that removed cascade would take both down.
///
/// So this ships 47 KB and the two instruments now agree on it, 47.6 against
/// 46.1, where before one of them was returning an impossibility. The
/// constant did not move; what moved is that it no longer rests on declaring
/// an instrument broken.
///
/// The honest label is still a lower bound, and the reason is now narrower
/// than it was. 47 KB is the part of a blocked tracker's subtree that lands
/// back on hosts some list names; the ad creatives and iframes on hosts no
/// list names are real and are not in it. The repaired whole-page instrument
/// is the only thing that could see them and it puts the gap at 1.5 KB with
/// an interval tens of KB wide -- so it has stopped contradicting the listed
/// figure without being able to add to it. Charging the 23 KB a request the
/// single-pass crawl implied is still not something this comment can justify.
/// See `Bounds.bytes_tracking_ratio_min` in `tests/top500.py` for what that
/// costs the test suite -- the sharp page-level bound now sits near 1.0
/// because both sides of it are the same quantity -- and
/// `Bounds.bytes_page_ratio_max_stable` for the bound that rests on the same
/// six-page filter as the agreement above.
///
/// That gap has since been measured directly rather than by subtraction, and
/// it stays unmeasured. `bytes_saved_unlisted_tp`, added to
/// `compare_tracking_arms.py`, counts the delta over third-party requests the
/// Disconnect list does *not* name -- exactly the missing part, and none of
/// the first-party video that makes the whole-page delta unreadable. Over the
/// ten passes it reads -3.02 MB a pass with a 725% one-pass standard
/// deviation and a 229% pooled standard error, swinging from -34.8 to
/// +35.6 MB; drift-corrected over the touched pages it comes to 0.2 KB a
/// request. The third-party video CDNs it still contains are enough to sink
/// it on their own. A narrower instrument did not rescue the quantity, so the
/// lower-bound label stands with no number on the gap.
///
/// Two more refinements were measured with a new instrument and neither is
/// shipped. `src/live_ablation.py` blocks *one* host on a live page and diffs
/// the paired totals, which prices a single tracker's subtree instead of
/// dividing a crawl-wide figure; `src/replay_ablation.py` does the same
/// against a recorded page, where the load is deterministic.
///
///   Memorising the cascade lost on every key, on the first run's data.
///   Looking the subtree up the way the byte estimate is looked up, instead
///   of answering with one number per context, was graded leave-one-out over
///   48 ablations on 21 pages -- predict a held-out cell from the other cells
///   sharing its key, against predicting it from the grand mean:
///
///   ```text
///   one constant       206.6 KB mean, 54.1 KB median absolute error
///   per host           253.5 KB mean, 94.9 KB median
///   per page site      282.3 KB mean, 126.3 KB median
///   per page category  273.0 KB mean, 155.5 KB median
///   ```
///
///   Two of those keys are still dead and the per-host one is not: pooled
///   over four runs on the listed width it wins, and ships as
///   [`FOLLOWUP_HOST_SCALE`], whose docstring gives the numbers and says what
///   was wrong with this grading. What survives unchanged is that *page*
///   identity is not a key at any width. The same host varies wildly between
///   pages on the third-party width --
///   `securepubads.g.doubleclick.net` reads 0.0, 10.4, 40.1, 311.5 and
///   425.9 KB a request on five, `connect.facebook.net` 1.6, -0.8, 0.5 and
///   211.3 on four -- but the page is no better a key than the host and in
///   fact worse, because the same page varies wildly too: nbcnews.com reads
///   19.2, 65.0 and 425.9 KB on its three, uol.com.br 311.5, -129.0 and
///   -274.6. So this is not "a property of the page rather than the request".
///   See `would_a_table_help` in `src/analyse_live_ablation.py`.
///
///   A page-level intercept alongside the per-request term is the third
///   explanation tried for a disagreement this constant now has with a direct
///   instrument, and it does not settle it either. Single-host ablation
///   (`src/live_ablation.py`) prices one host's subtree with the rest of the
///   page running, and over two 125-page runs it reads the listed cascade at
///   14.6 and 14.9 KB a request against this crawl's 47.0 -- both stable, so
///   they cannot both be per-request subtrees. Fitting a page term and a
///   request term together on the quiet instrument moves the per-request
///   figure to 33.2 KB and puts 29% of the mass on the page, which is the
///   predicted direction and a third of the distance; but the intercept's
///   95% interval is [0, 800] KB, the two-term form takes 77% of paired
///   bootstraps on per-page error against an 83% noise bar and loses on
///   category totals at 17%, and on the loud instrument the intercept is
///   exactly zero. The other two explanations fail more cleanly: the joint
///   arm reads 0.48 of the sum of the single-host arms, so blocked hosts
///   *over*count when summed rather than interacting upward, and restricting
///   the denominator to cascading aborts moves it 8%. See `intercept_fit` and
///   `grade_intercept` in `fit_followups.py`.
///
///   `src/sweep_ablation.py` then settled the intercept. It blocks a page's
///   ETP-blocked hosts cumulatively, k = 0..K in random order, so k = K is a
///   joint counterfactual and the curve in between is a dose response -- and
///   fitting `cascade(k) = a + b*k` on one page's own sweep identifies the
///   two terms that are collinear across pages. Over 44 pages the per-page
///   term's median is +0.2 KB at every width, against a per-host term of
///   +1.1 KB listed and +15.3 KB third-party, and the per-host term carries
///   69-95% of the mass. There is no page-level intercept; the cascade is per
///   request after all, which is what this array already assumed.
///
///   The same sweep is the first instrument here that can price the unlisted
///   half from a joint counterfactual. Blocking every host at once and
///   counting over the same denominator this array uses -- script and
///   document only, `CASCADING` in `fit_followups.py` -- it reads 18.9 KB a
///   request listed and 88.2 KB third-party, so about 69 KB a request lands
///   on hosts no list names. That is larger than the listed part rather than
///   a correction to it, and it is the first number of any kind on the gap
///   the lower-bound label describes. Two cautions keep it from moving this
///   constant. The sweep caps at six hosts a page and so reproduces 33% of
///   what ETP blocks, a partial joint counterfactual; and on the listed width
///   it reads 18.9 KB where the crawl reads 47.0, still 2.5x apart with the
///   denominators aligned, which coverage plausibly explains and no
///   measurement here demonstrates. Until that closes, 47 KB stays.
///
///   One thing did show up and did not survive being graded properly. What a
///   page memorises is not the magnitude but the *gate*: whether a blocked
///   tracker cascades at all is predicted from the page site at 73% against a
///   56% base rate, and a page's blocked-request count reproduces 69% of that
///   with no site identity at all, with the median cascade rising 10.2 ->
///   76.5 -> 87.4 KB across count buckets. That is 48 cells and several
///   hypotheses tried, so it was re-asked on the crawl's 7,184 cascading
///   blocks as a functional form, where it loses: see `grade_saturation`.
///
///   Consent managers do not reach this function at all. OneTrust, Osano and
///   Sourcepoint are on the Disconnect list and ablating one is dramatic --
///   +3.6 MB on braze.com, -634 KB on ted.com, reproducible to a few kB and
///   not the same sign, because blocking one changes the page's consent state
///   rather than pruning a subtree. But Firefox's ETP tracking tables are not
///   the Disconnect list: across all 18,891 blocked requests in the ten-pass
///   crawl, ETP blocked a consent manager zero times. See `cmp_cascade` in
///   `fit_followups.py`, which re-checks the count on every run, since what
///   makes this a non-problem is a fact about ETP's tables that could change.
///
/// HTTP Archive's 1% URL exports carry `initiator_type`, which is the request
/// tree's shape, over 9.7M tracker requests on 3.2M page domains. Roots
/// (parser-initiated) hold 12.6 GB against descendants (script-initiated) at
/// 58.5 GB, a ratio of 4.63, or 7.95 against script roots alone; counted per
/// root request rather than per root byte, that is 16 KB over all 3.6M roots
/// and 62 KB over the 947k script roots. Both are UPPER bounds rather than
/// measurements, and the reason is worth keeping in mind: the export says a
/// script asked for the request, not *which* script. Where the parent is
/// another tracker the child is cascade; where it is the page's own code, ETP
/// refuses the child directly and it belongs in the denominator instead.
/// Nothing available here separates the two.
///
/// HTTP Archive's 62 KB and the shipped 47 KB bound the same quantity, which
/// is the useful thing about them: HTTP Archive's numerator is tracker
/// requests, so its 62 KB is an upper bound on the part of a subtree that
/// lands back on listed hosts, and the crawl's 47 KB is a measurement of that
/// same part from the other direction. They bracket it, from two crawls, two
/// browsers and two months apart, to within a third. Neither says anything
/// about the rest, so a cascade-inclusive total should still be read as good
/// to a factor, not to two significant figures -- but it is now a factor in a
/// known direction: this understates.
///
/// Changing the form from per-byte to per-request, in 2026-09, did not move
/// the total: HTTP Archive's tracker scripts and documents average 31.8 KB,
/// so the 2.3 per byte this used to ship was already charging 73 KB to the
/// average one. What changed was the spread -- a 2 KB `gtag/js` stub was
/// charged 4.5 KB of subtree and a 106 KB `fbevents.js` was charged 244 KB,
/// where both are now charged the same 47 KB. Dropping 70 to 47 is the
/// separate, later change described above, and that one does move the total,
/// by a third.
///
/// The three constructions of the listed-only figure are worth keeping apart,
/// because two of them share almost nothing. 43.7 KB comes from the raw
/// page-level byte delta and 47.0 KB from the same delta with the untouched
/// pages' churn subtracted. 49.5 KB comes from the request *sets*: which URLs
/// the control arm asked for that the blocking arm never did, summed by size,
/// which never forms a difference of two page totals at all and is a lower
/// bound because a descendant both arms happened to request is invisible to
/// it. Two instruments and a bound, inside six percent.
///
/// HTTP Archive cannot supply that split, only the crawl can, and the reason
/// is the same one that makes the 4.63 above an upper bound: co-occurrence is
/// not causation. Fitting each host's contribution to a page's *unlisted*
/// third-party mass credits www.google-analytics.com, a 20 KB beacon script,
/// with 255 KB of it, and reads a listed share of 0.07 against the crawl's
/// 0.40-0.63. That is the fit saying heavy pages carry more of everything.
/// See `per_host_cascade_split`.
///
/// (An earlier revision of this comment quoted 2.18 from
/// `data/raw/per_request_1pct.csv`. That extract is not a uniform sample --
/// `sql/04_per_request_features.sql` keeps the 200 highest-variance and 50
/// most common tracker domains -- and on the uniform exports the same
/// quantity is 4.63. The shipped constant did not change; the evidence for it
/// did.)
///
/// Five refinements were tried against the crawl and none is shipped. All
/// five are in `scripts/fit_followups.py` with their numbers.
///
///   Per page category. Pages were hand-labelled into seven kinds
///   (`data/tranco_500_categories.csv`) on the theory that a news site's
///   cascade differs from a bank's. It very likely does, and this crawl
///   cannot see it: every category's 95% interval spans zero and runs into
///   the tens, because page variance swamps the effect. Entertainment pages
///   "measure" 12.5 on 20 pages. Shipping those would be shipping noise.
///
///   Per tracker category, from the Disconnect list. The most natural of the
///   four, since the list hands the category over for nothing and the story
///   writes itself: an Advertising tracker's subtree is an ad stack, an
///   Analytics tracker's is a beacon, a Social one's is a widget. The crawl
///   puts the first two in that order -- 36.7 KB and 27.5 KB a request
///   against a single 38.4 KB -- and cannot separate anything: every 95%
///   interval contains the single figure (Advertising [3, 64], Analytics
///   [0, 64], Social [0, 301]), and Social, the one the story is most
///   confident about, comes out highest of the three at 76 KB on a quarter
///   of the blocks.
///
///   This is where the ten-pass crawl paid for nothing, and the reason is
///   worth keeping, because an earlier revision of this comment predicted
///   the opposite in as many words: "a crawl ten times larger would separate
///   39 KB from 22 KB". The crawl is now ten times larger and separates
///   nothing -- the intervals are wider than they were, not narrower. Ten
///   passes over the same 271 pages remove the noise in how a *page load*
///   comes out and leave untouched the noise in which *pages* were crawled,
///   and the bootstrap here resamples pages. Repetition buys the page-level
///   denominators a great deal, because their problem is load-to-load churn;
///   it buys a per-category split nothing, because its problem is 271 pages.
///
///   The arithmetic of that is worth keeping too, because it is not about
///   Disconnect: 7,089 cascading blocks over 271 pages support one figure to
///   about a third of itself, so splitting it three ways widens each interval
///   by roughly the root of three while the categories differ by less than
///   that. What would pay is a crawl ten times *wider* -- 5,000 domains, one
///   pass each -- which is a different experiment from the one that was run.
///
///   Per tracker *role*, which is the same list category crossed with what
///   the request was for -- `ad-script`, `analytics-script`,
///   `social-script`, `ad-frame` and so on, the axis `tests/top500.py` now
///   bounds the *size* estimate on. It separates nothing here either:
///   ad-script 34.9 KB [0, 65], analytics-script 26.6 KB [0, 61],
///   social-script 76.0 KB [0, 301] against one figure of 38.4 KB, with
///   every interval containing it and most containing zero.
///
///   The contrast with what the same axis does for the size estimate is the
///   useful part, because it is not about the taxonomy at all. On size the
///   role split is sharp enough to find two real biases the crawl-wide
///   figure was cancelling, and on cascade it cannot separate anything. The
///   difference is the ground truth underneath: a size bound has 11,290
///   per-request observations, where a cascade bound has 271 pages, no
///   matter how many requests are on them. Splitting is affordable exactly
///   when the measurement is per request. See
///   `Bounds.max_tracker_role_bias_contribution_pct` in `tests/top500.py`.
///
///   Borrowing the shape from HTTP Archive instead fails the way the per-site
///   fit does, and more starkly: Analytics is on almost every page, so its
///   column doubles as the intercept and takes the whole cascade, leaving
///   Advertising -- 7.5M requests, the densest category in the export -- at
///   1.4 KB a request against its own 5.0 KB mean, hence no cascade at all.
///   Levelled and graded on the crawl that shape wins 8% of bootstraps on
///   per-page error against the quiet instrument and 6% against the loud one.
///
///   Also worth recording, since it is the one split that would cost no API
///   surface: SCRIPT against HTML, the axis `estimate_resources` already
///   carries. The crawl blocked 26 documents against 713 scripts, which gives
///   HTML an interval of [0, 182] KB and the two instruments opposite signs.
///
///   Per tracker site, by rootness. How often the markup asks for a host
///   directly is a plausible proxy for whether it is a loader or a leaf, and
///   it covers 90% of the crawl's cascading bytes. Reweighting by it --
///   levelled so it allocates the same total, so this is a question about
///   shape alone -- wins 66% of paired bootstraps on per-page error against
///   the whole-page delta and 48% on category totals.
///
///   Per tracker site, by fitted page share. The 50% export is dense enough
///   that a page keeps most of its requests, so each host's additive
///   contribution to a page's total tracker mass can be fitted directly:
///   1,826 hosts over 510k pages by non-negative least squares, with the
///   catch-all coefficient landing at 1.04 where 1.0 is exactly right. It is
///   the best-identified per-site shape available and it covers 87% of the
///   crawl's blocked requests. Against the whole-page delta it wins 27% of
///   bootstraps on per-page error and 57% on category totals.
///
/// Those two were once recorded here as coin flips, on the reading that the
/// crawl was too noisy to tell. Half of that was true and half was the wrong
/// instrument. Graded against the Disconnect-only delta instead -- four times
/// quieter, and the same split this comment's second paragraph rests on --
/// each wins 2% of paired bootstraps on per-page error against the shipped
/// model. On category totals they stay a coin flip: 41% and 43% of
/// bootstraps, with point estimates of 8.6 MB and 7.7 MB against the shipped
/// 9.0, which is a difference this crawl cannot call. The per-site magnitudes
/// are low by a factor of two besides.
///
/// So the finding is sharper than it was, and most of what those refinements
/// were reaching for was the *form*. A tracker site, and a tracker category
/// with it, is a proxy for what a script does with the network, and charging
/// the cascade per request rather than per byte captures that more cheaply
/// and more decisively than 1,826 coefficients or six categories did -- on
/// the same instrument, per request wins 89% where they win 2% and 8%. What would settle the residual question of whether a *given*
/// vendor still differs: a paired crawl an order of magnitude larger, or one
/// that records each blocked request's initiator, which Firefox has at block
/// time and `firefox_crawl_500_tracking.py` currently drops. Either would
/// also make a per-category factor fittable, and would settle the saturation
/// question above.
static FOLLOWUP_BYTES_PER_REQUEST: [i64; 11] = [
    47_000, // SCRIPT  loaders, tag managers, ad stacks: the whole of the effect
    0,      // IMAGE   a pixel fetches nothing
    0,      // VIDEO
    0,      // OTHER   beacons and XHR payloads; the response is data, not code
    0,      // AUDIO
    0,      // CSS     can pull fonts and images; unmeasured, see above
    0,      // FONT
    47_000, // HTML    an ad or social iframe, which is a page of its own
    0,      // TEXT
    0,      // WASM    it runs, but it is fetched *by* a script already counted
    0,      // XML
];

/// The cascade charged to the *first* block on a host in a page load, when the
/// caller can tell [`followup_bytes_for`] which those are.
///
/// Indexed like [`FOLLOWUP_BYTES_PER_REQUEST`], and the same quantity divided
/// by a better denominator. That array spreads the measured cascade over every
/// cascading blocked request; this one spreads it over the distinct hosts
/// those requests went to, because that is what a pruned subtree corresponds
/// to. A page where ETP blocks twelve `doubleclick.net` requests lost one ad
/// stack, not twelve.
///
/// THE MEASUREMENT. Over the crawl's 271 touched pages the listed-only delta
/// leaves 333.4 MB of cascade against 7,089 cascading blocks -- 47.0 KB a
/// request, which is `FOLLOWUP_BYTES_PER_REQUEST` -- and against 5,441
/// distinct host-page-load triples, which is 61.3 KB, 95% [42, 84]. The two
/// describe the same 333.4 MB; they differ only in how it is spread.
///
/// THE EVIDENCE THAT THE SPREADING IS BETTER, which is the part that matters,
/// because both forms allocate the same total by construction. Graded the way
/// `scripts/fit_followups.py` grades every other refinement -- levelled to a
/// fixed crawl total, paired bootstrap over pages -- charging per distinct
/// host beats charging per request on the quiet Disconnect-only instrument by
/// 99% of bootstraps on per-page error (433.8 MB down to 417.9) and 85% on
/// category totals (70.8 MB down to 56.2). It is the first refinement tried
/// against this crawl to clear the file's 83% bar on *both* metrics; the
/// per-request form itself clears it on neither, at 82% and 59%. The loud
/// whole-page delta agrees on per-page error at 83% and does not on category
/// totals at 42%, which is the dissent every shape in this file draws from it.
///
/// AND THAT IT IS THE HOSTS DOING THE WORK. A model that charges fewer units
/// on heavily-blocked pages could win by being a saturation term in disguise,
/// and saturation loses here (see `grade_saturation`). The control is a
/// permutation: keep each page's count of cascading blocks per pass and draw
/// their hosts at random from the crawl-wide pool, so the counts collapse by
/// the same amount for no reason. Over 100 shuffles that wins about 48% of
/// bootstraps on average, none of them reaches the real 99%, and the
/// shuffled model's per-page error comes back to the flat model's to within
/// 0.2 MB. The effect is which hosts, not how many requests.
///
/// WHAT IT COSTS A CALLER. Knowing, per page load, which hosts have already
/// had a cascading tracker blocked. Firefox has that at block time; a caller
/// scoring a log row by row does not, and passes [`CascadeRoot::UNKNOWN`] to
/// get the average instead. Both are supported and neither is a guess.
///
/// Repeats are charged zero rather than a fraction. A discount between the
/// two grades slightly better on consistency and worse on effect -- at half
/// weight the per-page error falls to 425.9 MB against 417.9 at zero -- and
/// zero is what the model actually claims: the subtree was already counted.
///
/// WHAT IT ABSORBS, which is the part that makes it look like a rule rather
/// than a fitted correction. Every refinement this crate rejects was
/// re-graded on top of this one, and the count-based cousin of it collapses:
/// charging `hosts ** gamma` instead of `hosts` now wins 11% to 32% of
/// bootstraps against the 19% to 57% it managed against the per-request
/// form. Saturation was always this, measured through the wrong variable --
/// "two loaders share an ad stack" is a fact about which hosts they are, and
/// once the hosts are counted there is nothing left in the count. Scaling by
/// the blocked request's own size does better on this baseline than on the
/// old one, 71% against 24%, and still does not reach the 83% bar.
///
/// 61 KB is robust to the one page filter this repo applies: over the 265
/// pages whose loads repeat it reads 60.2 KB. See `page_delta_instability`
/// in `tests/top500.py`.
///
/// The grain is the full hostname and not the registrable domain, by
/// measurement rather than by taste -- the coarser grain loses at 28%. A
/// vendor's loader and its beacon endpoint live on different hostnames and
/// prune different subtrees.
///
/// HTML is given the same figure as SCRIPT here for the same reason it is in
/// [`FOLLOWUP_BYTES_PER_REQUEST`] and no better one: 199 of the 5,441
/// host-firsts are documents, which is too few to separate.
/// Re-derived from 61 KB when [`FOLLOWUP_HOST_SCALE`] shipped, and the reason
/// is worth stating because the number looks like a fudge otherwise. That
/// shape is levelled over the crawl's blocked *requests*, which is the
/// population [`FOLLOWUP_BYTES_PER_REQUEST`] is charged over. The `FIRST`
/// path charges over distinct host-page-loads instead, and the two
/// populations have different mixes -- a host blocked eight times on a page
/// carries eight times the weight in one and the same weight in the other --
/// so a shape levelled for one is 1.9% off for the other. Dividing by that
/// keeps the two constants what they are meant to be: one measurement over
/// two denominators, which `test_charging_the_cascade_per_host_preserves_the_total`
/// asserts to 1%.
static FOLLOWUP_BYTES_PER_HOST: [i64; 11] = [
    59_900, // SCRIPT  as above, per distinct host rather than per request
    0,      // IMAGE
    0,      // VIDEO
    0,      // OTHER
    0,      // AUDIO
    0,      // CSS
    0,      // FONT
    59_900, // HTML
    0,      // TEXT
    0,      // WASM
    0,      // XML
];

/// Per-host modulation of the cascade, looked up by hostname suffix.
///
/// Dimensionless and levelled: the entries average to 1.0 over the crawl's
/// cascading blocks, so this redistributes the cascade between hosts without
/// changing what the estimator claims in total. A host no entry matches is
/// charged 1.0, which is every host on the open web bar the ones measured
/// here.
///
/// MATCHING IS BY SUFFIX, MOST SPECIFIC FIRST. `followup_host_scale` walks
/// `a.b.example.com`, `b.example.com`, `example.com` and stops before the
/// bare TLD, taking the first key that hits. So one `google-analytics.com`
/// entry prices `www.`, `region1.` and every other regional shard that was
/// never crawled, and one `sentry.io` entry prices the eleven per-customer
/// `o<digits>.ingest.` hosts in the sample plus every customer who was not.
///
/// Exact hosts are kept only where they disagree with their suffix by more
/// than a quarter of the average cascade, which is both the accuracy and the
/// compression. `doubleclick.net` needs three entries because it holds
/// `securepubads` at 4.00 against `ad` and `googleads` at 0.14;
/// `google-analytics.com` needs one because its two measured hosts read -2
/// and +2 KB. 83 suffix entries and 9 exact overrides cover 115 measured
/// hosts.
///
/// Neither flat form is as good. Graded leave-one-out over the pooled cells,
/// coarsening everything to the suffix reads 45.2 KB against exact-host
/// matching's 40.2, because a suffix cannot hold `doubleclick.net`'s spread;
/// exact-host matching reads 40.2 but generalises to nothing, since a
/// regional shard or a customer id is never seen twice. Both together read
/// 41.1, within noise of the better one, and cover the hosts neither run saw.
/// The single constant reads 55.7.
///
/// KEYS ARE HASHED, NOT STORED. Each entry is the FNV-1a/32 of its key --
/// [`Fnv1a`] with ASCII lowercasing, the same hash and folding the main table
/// uses, reproduced in `fit_host_cascade.py` -- beside a `u8` scale read as
/// `q / 63.75`. Five bytes an entry against roughly fifty for a `(&str, f64)`
/// pair, no string comparison in the lookup, and no hostnames in the binary.
/// The cost is that a collision would misprice a host silently, so
/// `host_key_table_distinct` asserts the shipped keys are distinct and
/// `host_scale_matches_generator` pins the hash against known values.
///
/// WHAT IS MEASURED. Single-host ablation on live pages: block one tracker
/// host, load the page paired against a placebo arm, and read the delta over
/// Disconnect-matched requests less the host's own bytes. 646 (run, page,
/// host) cells over four runs and 115 hosts. See
/// `llm-classifier/scripts/fit_host_cascade.py` and `src/live_ablation.py`.
///
/// WHY THIS PER-HOST FIT SHIPS WHEN THE OTHER PER-SITE ONES DID NOT. Three
/// were rejected above, including a per-host table graded on a single 21-page
/// run, which lost at 253.5 KB against the constant's 206.6. That grading
/// used the third-party width, where per-impression ad creatives dominate and
/// nothing is a property of the host; it had 48 cells; and it never checked
/// whether between-host signal existed. On the listed width, pooled over four
/// runs, it does: the per-cell correlation between two independent 125-page
/// runs is 0.79, their means agree to 4%, and the between-host variance
/// exceeds the within-host sampling variance.
///
/// Shrinking each host towards the grand mean was tried at the fitted
/// k = 0.56 and is worse than not shrinking, on rarely-seen hosts (24.4
/// against 30.4) as much as common ones, so these are plain means behind a
/// two-observation floor.
///
/// WHAT IT IS NOT. It is not a magnitude. The sweep instrument reads 18.9 KB
/// a request where the crawl reads 47.0 on the same width with denominators
/// aligned, and 88.2 KB once unlisted bytes are counted; 646 observations do
/// not settle that and a levelled shape does not have to. Entries are clamped
/// to [0.10, 4.00] and re-levelled until both hold, so no one host can carry
/// the table.
/// FNV-1a/32 of each key, ascending. 92 entries, 460 bytes with [`FOLLOWUP_HOST_Q`].
static FOLLOWUP_HOST_KEY: [u32; 92] = [
    0x019f7906, 0x0204997c, 0x02d67ea1, 0x03738b92, 0x0b2922c4,
    0x0f029a53, 0x0f5d12ff, 0x105783b5, 0x141654bf, 0x1560c440,
    0x15a86b59, 0x193165cd, 0x1a28b50f, 0x1e6afd91, 0x26dab89e,
    0x2b4fa416, 0x33bd440f, 0x35600496, 0x3613f233, 0x3a453640,
    0x3feef327, 0x4e717b32, 0x585cb01b, 0x58cbf7bc, 0x59a5b12a,
    0x5a5bb14a, 0x617fe839, 0x646b899b, 0x68c8c91a, 0x6a5bd546,
    0x6a680031, 0x6c3e32e9, 0x78ae9fb3, 0x79c51452, 0x7c7c9f90,
    0x7dabaa9b, 0x7e98e1ae, 0x7fa9782d, 0x8054ed5c, 0x83318cfe,
    0x852b6315, 0x8763d943, 0x8aeec33e, 0x8fc9f303, 0x905bc57d,
    0x91c7bec9, 0x94598838, 0x96e965aa, 0x99e55910, 0x9a553b82,
    0x9b2c0bf4, 0x9c3fc9fe, 0x9fb0cbe9, 0xa20d64d9, 0xa27d8ae0,
    0xa6e828ac, 0xa882dca7, 0xa8b08fb0, 0xb14762d5, 0xb865c83b,
    0xbaef443c, 0xbe4e1219, 0xc5eea928, 0xc8079ede, 0xc85dde1d,
    0xce232721, 0xcefb7100, 0xd0a600cd, 0xd206f790, 0xd288b58b,
    0xd6369852, 0xd6f47d10, 0xd8a62f82, 0xd97e1eeb, 0xd9c05c45,
    0xded59275, 0xdf11b043, 0xdfa600a7, 0xe0d598a7, 0xe8388bfe,
    0xe969600a, 0xedaf73f6, 0xeea003e0, 0xef6b78bd, 0xefe21e8b,
    0xf1079036, 0xf2a23a20, 0xf8e64dd3, 0xfacf015c, 0xfccdb5f2,
    0xfd052483, 0xfdea922b,
];

/// Scale for the key at the same index, as `q / 63.75`.
static FOLLOWUP_HOST_Q: [u8; 92] = [
    9, 9, 9, 72, 12, 251, 9, 251, 9, 9, 9, 9, 251, 10, 9, 251, 9, 9, 9,
    16, 9, 9, 9, 9, 9, 9, 9, 9, 9, 113, 9, 251, 9, 9, 9, 251, 9, 9, 9, 9,
    9, 251, 174, 251, 177, 9, 9, 9, 9, 15, 251, 90, 9, 9, 9, 9, 9, 77, 32,
    9, 9, 9, 9, 251, 14, 9, 251, 9, 46, 9, 137, 9, 9, 9, 44, 9, 251, 9,
    32, 9, 9, 32, 9, 251, 9, 9, 155, 130, 9, 9, 251, 251,
];

// The keys above, in the same order, so the table can be read:
//
//   0x019f7906     client.aps.amazon-adsystem.com               0.13
//   0x0204997c  *.omtrdc.net                                   0.13
//   0x02d67ea1  *.aticdn.net                                   0.13
//   0x03738b92  *.useinsider.com                               1.12
//   0x0b2922c4  *.sc-static.net                                0.19
//   0x0f029a53     securepubads.g.doubleclick.net               3.93
//   0x0f5d12ff  *.optimizely.com                               0.13
//   0x105783b5  *.media.net                                    3.93
//   0x141654bf  *.crazyegg.com                                 0.13
//   0x1560c440     ad.doubleclick.net                           0.13
//   0x15a86b59  *.survicate.com                                0.13
//   0x193165cd  *.tvpixel.com                                  0.13
//   0x1a28b50f  *.scorecardresearch.com                        3.93
//   0x1e6afd91  *.quantummetric.com                            0.16
//   0x26dab89e  *.tiktok.com                                   0.13
//   0x2b4fa416  *.ensighten.com                                3.93
//   0x33bd440f  *.webcontentassessor.com                       0.13
//   0x35600496  *.sail-personalize.com                         0.13
//   0x3613f233  *.frugalfiestas.com                            0.13
//   0x3a453640  *.trustpilot.com                               0.25
//   0x3feef327  *.privacymanager.io                            0.13
//   0x4e717b32  *.perfdrive.com                                0.13
//   0x585cb01b  *.addtoany.com                                 0.13
//   0x58cbf7bc  *.sentry-cdn.com                               0.13
//   0x59a5b12a  *.webvisor.org                                 0.13
//   0x5a5bb14a  *.tsyndicate.com                               0.13
//   0x617fe839  *.posthog.com                                  0.13
//   0x646b899b  *.liadm.com                                    0.13
//   0x68c8c91a  *.mrf.io                                       0.13
//   0x6a5bd546  *.facebook.net                                 1.76
//   0x6a680031  *.go-mpulse.net                                0.13
//   0x6c3e32e9  *.servenobid.com                               3.93
//   0x78ae9fb3  *.google-analytics.com                         0.13
//   0x79c51452  *.bing.com                                     0.13
//   0x7c7c9f90  *.demdex.net                                   0.13
//   0x7dabaa9b  *.licdn.com                                    3.93
//   0x7e98e1ae  *.htlbid.com                                   0.13
//   0x7fa9782d  *.mxpnl.com                                    0.13
//   0x8054ed5c  *.mail.ru                                      0.13
//   0x83318cfe  *.inspectlet.com                               0.13
//   0x852b6315  *.cloudflareinsights.com                       0.13
//   0x8763d943  *.hs-analytics.net                             3.93
//   0x8aeec33e     s.amazon-adsystem.com                        2.73
//   0x8fc9f303     js.qualified.com                             3.93
//   0x905bc57d  *.adtrafficquality.google                      2.78
//   0x91c7bec9  *.2mdn.net                                     0.13
//   0x94598838  *.siteimproveanalytics.com                     0.13
//   0x96e965aa  *.amplitude.com                                0.13
//   0x99e55910  *.googleadservices.com                         0.13
//   0x9a553b82  *.mts.ru                                       0.23
//   0x9b2c0bf4  *.permutive.app                                3.93
//   0x9c3fc9fe  *.clarity.ms                                   1.42
//   0x9fb0cbe9  *.hs-scripts.com                               0.13
//   0xa20d64d9  *.flocktory.com                                0.13
//   0xa27d8ae0  *.chartbeat.com                                0.13
//   0xa6e828ac  *.quicklyedit.com                              0.13
//   0xa882dca7  *.lightboxcdn.com                              0.13
//   0xa8b08fb0  *.btloader.com                                 1.21
//   0xb14762d5     mc.yandex.ru                                 0.50
//   0xb865c83b  *.ssl-images-amazon.com                        0.13
//   0xbaef443c  *.contentsquare.net                            0.13
//   0xbe4e1219  *.redditstatic.com                             0.13
//   0xc5eea928  *.appboycdn.com                                0.13
//   0xc8079ede     js.hs-analytics.net                          3.93
//   0xc85dde1d  *.sail-horizon.com                             0.22
//   0xce232721  *.yandex.ru                                    0.13
//   0xcefb7100  *.doubleclick.net                              3.93
//   0xd0a600cd  *.browser-intake-datadoghq.com                 0.13
//   0xd206f790  *.googlesyndication.com                        0.73
//   0xd288b58b  *.trafficjunky.net                             0.13
//   0xd6369852  *.adroll.com                                   2.15
//   0xd6f47d10     googleads.g.doubleclick.net                  0.13
//   0xd8a62f82  *.adsrvr.org                                   0.13
//   0xd97e1eeb  *.google.com                                   0.13
//   0xd9c05c45  *.vk.com                                       0.69
//   0xded59275  *.nr-data.net                                  0.13
//   0xdf11b043     app.qualified.com                            3.93
//   0xdfa600a7  *.magsrv.com                                   0.13
//   0xe0d598a7  *.yandex.com                                   0.49
//   0xe8388bfe  *.newrelic.com                                 0.13
//   0xe969600a  *.sensic.net                                   0.13
//   0xedaf73f6  *.hs-banner.com                                0.51
//   0xeea003e0  *.segment.com                                  0.13
//   0xef6b78bd  *.qualified.com                                3.93
//   0xefe21e8b  *.parsely.com                                  0.13
//   0xf1079036  *.sentry.io                                    0.13
//   0xf2a23a20  *.visualwebsiteoptimizer.com                   2.43
//   0xf8e64dd3  *.amazon-adsystem.com                          2.04
//   0xfacf015c  *.zi-scripts.com                               0.13
//   0xfccdb5f2  *.hubspot.com                                  0.13
//   0xfd052483  *.rambler.ru                                   3.93
//   0xfdea922b  *.taboola.com                                  3.93

/// Per-rung modulation of the cascade, indexed by [`UrlMatch`] discriminant.
///
/// Dimensionless and levelled: the magnitude lives in
/// [`FOLLOWUP_BYTES_PER_REQUEST`] and is unchanged by this, because over the
/// crawl's mix of rungs these average to 0.999. Shape only. That is the
/// safeguard that makes shipping them defensible, and the rest of this
/// comment is about why they need one.
///
/// WHAT IS MEASURED. Fit a cascade per rung on the paired crawl -- one row
/// per page, counts of cascading blocks by rung against what the page shed
/// -- and the rungs come out in the same order on both of the crawl's quiet
/// instruments. Against the request-set difference: `host_template`
/// 27.0 KB, `prefix` 43.9, `ext_query` 75.4. Against the Disconnect-only
/// byte delta, which shares nothing with it but the design matrix: 32.5,
/// 41.9, 80.4. Two constructions agreeing to about 20% on three
/// coefficients is the positive evidence here, and there is a mechanism
/// that fits it -- `prefix` matches the stable path families of ad stacks,
/// `host_template` the per-customer subdomains of single-purpose analytics
/// endpoints, and `ext_query` is a script the table cannot place at all.
///
/// WHY THE FACTORS BELOW ARE NOT THOSE NUMBERS. Because the agreement is
/// weaker than it looks and the third instrument dissents. Resampling
/// pages, P(`prefix` > `host_template`) is 78% and 64% on the two quiet
/// instruments; P(`ext_query` > `prefix`) is 76% and 80% on those and 17%
/// on the loud whole-page delta, which orders them the other way. The
/// comment on the refinements this file *rejects* treats 83% as noise, and
/// nothing here clears that bar.
///
/// So the fitted shape is shrunk toward the single figure by how much of it
/// survives its own noise -- the standard answer to "the point estimates
/// differ and the intervals overlap". The between-rung standard deviation
/// that survives is 5.9 KB against a mean standard error of 27.3 KB, giving
/// per-rung weights of 0.01 to 0.31, so everything collapses most of the
/// way to the middle and what is left is the +-6% below. Read that as the
/// honest width of the finding rather than as a hedge: the crawl supports a
/// modulation of about a sixteenth, and this ships a sixteenth.
///
/// Then levelled, which is what makes shipping a weak shape safe at all: if
/// the ordering is wholly noise, the cost is that some requests are priced
/// 6% high and others 6% low, and no total moves.
/// `Bounds.max_cascade_levelling_pct` in `tests/top500.py` asserts that
/// property directly rather than trusting this comment, and
/// `scripts/fit_followups.py` reproduces every number above, dissent
/// included, and prints these factors.
///
/// ONE PLACE THESE ARE APPLIED AND NOT EVIDENCED. The fit above counts
/// blocked requests per rung, so the factors are identified on the per-request
/// denominator -- the one [`CascadeRoot::UNKNOWN`] uses. Refit on the
/// per-distinct-host denominator [`CascadeRoot::FIRST`] uses, the same
/// derivation returns a between-rung standard deviation of 0.0 KB against a
/// mean standard error of 39.8 KB, so every weight is zero and every factor
/// is 1.00: there are 5,441 units rather than 7,089 and the spread no longer
/// clears its own noise. The ordering survives -- 44.6, 57.7 and 82.8 KB for
/// `host_template`, `prefix` and `ext_query`, the same order as above -- but
/// the evidence for modulating by it does not.
///
/// [`followup_bytes_for`] applies these on both paths anyway, and that is a
/// choice worth naming rather than an oversight. Graded head to head on the
/// per-host path they are a wash: 44% of bootstraps on per-page error and 66%
/// on category totals, against a baseline of applying nothing. They are
/// levelled, so no total moves either way. Keeping them costs nothing
/// measurable and keeps the two paths differing only in their constant, which
/// is what makes `max_cascade_rehoming_pct` a meaningful check. Dropping them
/// on the `FIRST` path would be equally defensible and slightly more honest;
/// what would settle it is a crawl large enough to identify the shape on
/// 5,441 units.
static FOLLOWUP_RUNG_SCALE: [f64; 9] = [
    1.07, // Path          70 blocks; no evidence of its own -- see below
    0.95, // Template      331
    1.05, // Prefix        2,968 -- ad stacks with stable path families
    1.07, // Host          20
    0.94, // HostTemplate  3,149 -- per-customer analytics subdomains
    1.09, // ExtQuery      541 -- a script the table cannot place
    1.07, // Ext           10
    1.07, // Context       0 on this crawl
    1.00, // Bodyless      charged no cascade at all, so never read
];

// The four rungs with under 100 cascading blocks between them carry 1.07
// rather than 1.00, and it is not a claim about them. The joint fit sits
// below `FOLLOWUP_BYTES_PER_REQUEST` in level -- it is a different
// construction from the per-request division that set the constant -- so
// levelling multiplies every factor by about 1.07, and a rung with no
// evidence keeps that multiplier and nothing else. Their shrinkage weight
// is 0.01.


/// Main-thread milliseconds per KiB of follow-up transfer.
///
/// The follow-ups are charged at a blended rate rather than at the rate of
/// whatever blocked them, because they are not the same kind of resource as
/// their parent. HTTP Archive says what they are: over 9.7M tracker requests,
/// the script-initiated bytes are 88.4% scripts, 6.0% documents, 3.6% JSON
/// and 0.7% text. Weighting [`CPU_MS_PER_KIB`] by that mix gives
/// 1.8 ms/KiB.
///
/// This ships 0.9, half of it, and the reason it was set there was a
/// measurement: on the single-pass crawl, pages shed 92.3 s of CPU against
/// the 46.0 s these coefficients charge for the blocked requests themselves,
/// leaving 46.3 s for 52.4 MB of follow-up bytes, or 0.90 ms/KiB. The
/// composition said 1.8, the crawl said 0.89, and the gap was read as
/// evidence that [`CPU_MS_PER_KIB`] -- provisional, unfitted, anchored on a
/// mid-tier mobile parse rate -- is high by about a factor of two.
///
/// The ten-pass crawl re-ran that arithmetic and got 1.11 ms/KiB: 79.7 s shed
/// per pass against 43.7 s charged directly, leaving 36.0 s for 33.3 MB. The
/// constant did not move, and the reason is that the new figure is the first
/// one that comes with an error bar large enough to see. Pooling the passes
/// measures the CPU saving's standard error directly, at 54% -- so the 36.0 s
/// numerator is 36 ± 43 and the rate it implies is 1.11 ± 1.32. The
/// measurement cannot tell 0.9 from 1.1, or either from zero.
///
/// It is worth being blunt about why this instrument got *worse* when the
/// byte one got better. The ten-pass crawl runs its whole control arm before
/// its whole blocking arm, so a page's two loads are 133 minutes apart. Byte
/// counts do not care much; a machine's CPU does. The raw CPU delta over the
/// pages ETP touched is -89 s per pass -- the blocking arm burned *more* --
/// and only the drift correction, worth -6.2 s per untouched page, turns
/// that into +79.7 s. A correction an order of magnitude larger than the
/// signal is not a measurement, and `Bounds.cpu_ratio_min` says so in the
/// width of its band. A crawl that interleaved its arms would fix this and
/// nothing else here would need to change.
///
/// At 0.9 a cascade-inclusive CPU total comes to 73.0 s per pass against the
/// 79.7 s measured. That is still close enough to the figure the constant was
/// set from that `tests/top500.py` scores its CPU bands on the direct
/// estimate instead: a ratio near 1 against your own calibration target is a
/// tautology, not a test. See `Bounds.cpu_ratio_max`.
const FOLLOWUP_CPU_MS_PER_KIB: f64 = 0.9;

/// Follow-up transfer for a request in `context` whose own estimate is `bytes`.
///
/// `bytes` no longer scales the answer -- see
/// [`FOLLOWUP_BYTES_PER_REQUEST`] -- but it still gates it. An estimate of
/// zero says the table expects no body, and a response with no body runs no
/// code and fetches nothing on its way; charging it a subtree would be
/// charging one to something that never executed.
///
/// `matched` is how specifically the table recognised the URL, and it
/// modulates the answer by [`FOLLOWUP_RUNG_SCALE`] -- +-8%, levelled, so it
/// redistributes the cascade between rungs without changing what the
/// estimator claims in total. It is not free of the caller: the same walk
/// that produced `bytes` produced it, which is why [`resolve`] returns both
/// and [`estimate_resources`] no longer goes through [`estimate`].
/// The multiplier for `host`, or 1.0 when no key matches.
///
/// Walks suffixes most specific first and stops before the bare TLD, so
/// `a.b.example.com` tries three keys and `example.com` one. See
/// [`FOLLOWUP_HOST_SCALE`] -- the doc above the key array -- for why the
/// grain is a suffix and why the keys are hashes.
fn followup_host_scale(host: &str) -> f64 {
    let mut key = host;
    loop {
        // Two labels is the shortest key the generator emits, so a candidate
        // with no dot left is the bare TLD and there is nothing more to try.
        if !key.contains('.') {
            return 1.0;
        }
        let mut h = Fnv1a::new();
        h.write_cased(key.as_bytes(), true);
        let folded = h.fold();
        if let Ok(i) = FOLLOWUP_HOST_KEY.binary_search(&folded) {
            return f64::from(FOLLOWUP_HOST_Q[i]) / 63.75;
        }
        match key.find('.') {
            Some(i) => key = &key[i + 1..],
            None => return 1.0,
        }
    }
}

fn followup_bytes_for(bytes: i64, context: RequestContext, matched: UrlMatch,
                      root: CascadeRoot, host: &str) -> i64 {
    if bytes <= 0 {
        return 0;
    }
    // Indexing is in range because every array here has one slot per variant.
    let base = match root {
        // The subtree this request sits in was charged to an earlier block on
        // the same host, and charging it again would count it twice.
        CascadeRoot::REPEAT => return 0,
        CascadeRoot::FIRST => FOLLOWUP_BYTES_PER_HOST[context as usize] as f64,
        CascadeRoot::UNKNOWN => FOLLOWUP_BYTES_PER_REQUEST[context as usize] as f64,
    };
    (base * FOLLOWUP_RUNG_SCALE[matched as usize] * followup_host_scale(host))
        .round() as i64
}

/// CPU the follow-up transfer would have cost. See [`FOLLOWUP_CPU_MS_PER_KIB`].
///
/// No per-request term, unlike [`cpu_ms_for`]: the follow-ups are bytes here,
/// not requests, and nothing in this model says how many requests they arrive
/// in. A zero-byte follow-up estimate is therefore zero CPU, which is right --
/// it means no follow-up was predicted at all.
fn followup_cpu_ms_for(followup_bytes: i64) -> f64 {
    (followup_bytes.max(0) as f64 / 1024.0) * FOLLOWUP_CPU_MS_PER_KIB
}

/// The estimator proper, kept free of Python types.
///
/// `_initiator` is accepted and ignored; see [`RequestInitiator`].
/// How specifically the table recognised a URL: [`classify`]'s answer.
///
/// The estimator walks [`Level`] from the exact path down to the file
/// extension and answers from the first rung that has an entry. Which rung
/// that was is a fact about the URL that the lookup computes anyway and used
/// to discard, and it is worth more than it looks: it is the estimator's own
/// statement of how much it knew before it answered. `Path` means the table
/// has seen this exact asset; `Ext` means it knew nothing but the file type
/// and returned the mean of every `.js` in HTTP Archive.
///
/// It is a *classification of the URL* and not of the response, which is what
/// makes it usable at block time -- Firefox has the URL and the context and
/// nothing else -- and what makes it a fair axis to grade the estimator on.
/// Two other candidates were not: the Disconnect list's category needs a
/// second lookup the caller may not have, and a hand-written role taxonomy
/// (loader, beacon, pixel) would have been drawn by a human who had already
/// seen which URLs the estimator gets wrong. This one is decided by
/// `scripts/build_table.py` out of HTTP Archive, months before any crawl it
/// is graded against, and nothing about this crawl can move it.
///
/// `tests/top500.py` cuts its byte bounds by this; see
/// `Bounds.max_match_level_bias_contribution_pct` there for what the cut
/// found, which is that the rungs are not equally calibrated and the
/// aggregate hides it.
/// The discriminants index [`FOLLOWUP_RUNG_SCALE`], so reordering these
/// silently re-assigns cascade factors. Adding a variant means adding a slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UrlMatch {
    /// The exact path matched: [`Level::Path`].
    Path = 0,
    /// The path with versions and hashes collapsed: [`Level::Template`].
    Template = 1,
    /// The asset family, two templated segments and the extension.
    Prefix = 2,
    /// The host, but no path rung: the vendor is known, the asset is not.
    Host = 3,
    /// The host with generated labels collapsed.
    HostTemplate = 4,
    /// Extension and query length only.
    ExtQuery = 5,
    /// Extension only: the coarsest rung that is still a table entry.
    Ext = 6,
    /// No rung matched. Answered from `table::CONTEXT_FALLBACK`, the mean
    /// over a whole request context.
    Context = 7,
    /// The method forbids a response body, so no lookup happened at all and
    /// the URL was never consulted.
    Bodyless = 8,
}

impl UrlMatch {
    /// A stable lower-case name, for callers across an FFI boundary.
    ///
    /// The Python extension returns these rather than an enum because a
    /// `#[pyclass]` costs about 18 KiB of type-object machinery and the
    /// extension is held to a size budget; see the `llm-classifier-python`
    /// module docs.
    pub fn name(self) -> &'static str {
        match self {
            UrlMatch::Path => "path",
            UrlMatch::Template => "template",
            UrlMatch::Prefix => "prefix",
            UrlMatch::Host => "host",
            UrlMatch::HostTemplate => "host_template",
            UrlMatch::ExtQuery => "ext_query",
            UrlMatch::Ext => "ext",
            UrlMatch::Context => "context",
            UrlMatch::Bodyless => "bodyless",
        }
    }
}

impl From<Level> for UrlMatch {
    fn from(level: Level) -> Self {
        match level {
            Level::Path => UrlMatch::Path,
            Level::Template => UrlMatch::Template,
            Level::Prefix => UrlMatch::Prefix,
            Level::Host => UrlMatch::Host,
            Level::HostTemplate => UrlMatch::HostTemplate,
            Level::ExtQuery => UrlMatch::ExtQuery,
            Level::Ext => UrlMatch::Ext,
        }
    }
}

/// How specifically the table recognises `url`, without pricing it.
///
/// The same walk [`estimate_resources`] does, reporting the rung instead of
/// the value, so the two cannot disagree about which one answered: both call
/// [`resolve`]. Takes the same arguments for the same reason -- the context
/// is part of every lookup key, so the same URL can match at different rungs
/// as a script and as an image, and a bodyless method skips the walk.
pub fn classify(
    url: &str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: &str,
) -> UrlMatch {
    resolve(url, context, initiator, method).1
}

/// The size estimate and the rung it came from, in one walk of the table.
///
/// `#[inline(never)]` is load-bearing and is about the extension's size
/// budget rather than about speed. This walk is the bulk of the crate's code
/// -- `UrlParts`, both templates, the FNV hashing and the unrolled rung loop
/// -- and it has two public callers. Left to itself LLVM inlines it into
/// each, which costs 16 KiB of duplicate machine code in
/// `llm-classifier-python`, a third of that crate's whole allowance over the
/// baseline extension. One shared copy costs one call. See
/// `test_llmclassifier_size` in `tests/test_browsing_journey.py`.
#[inline(never)]
fn resolve(
    url: &str,
    context: RequestContext,
    _initiator: RequestInitiator,
    method: &str,
) -> (i64, UrlMatch) {
    // A response to one of these carries no payload, so the URL has nothing to
    // say about its size and no level of the table is worth consulting. The
    // table would otherwise answer a preflight from the path it shares with
    // the POST it precedes, which is the error this exists to avoid. See
    // BODYLESS_METHODS in scripts/build_table.py.
    if is_bodyless(method) {
        return (dequantize(table::BODYLESS_MEAN), UrlMatch::Bodyless);
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
            return (dequantize(value), level.into());
        }
    }
    // Nothing in the URL matched, so the context is all that is left to answer
    // from -- and it answers on its own terms: a font and a video have little
    // in common, and one mean over both is a worse guess than either.
    // Indexing is in range because the array has one slot per variant.
    (dequantize(table::CONTEXT_FALLBACK[ctx]), UrlMatch::Context)
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

// --------------------------------------------------------------------------- //
// Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// `followup_host_scale` binary-searches, so an out-of-order key would
    /// not fail loudly -- it would silently return 1.0 for a host that has a
    /// measurement. `fit_host_cascade.py` emits ascending; this is the guard
    /// that a hand edit cannot quietly undo it.
    #[test]
    fn host_key_table_sorted() {
        for pair in FOLLOWUP_HOST_KEY.windows(2) {
            assert!(pair[0] < pair[1], "not ascending at {:#010x}", pair[0]);
        }
    }

    /// Hashing the keys buys five bytes an entry and costs the guarantee that
    /// two keys are different. Distinctness follows from sortedness above, but
    /// assert it separately so the reason is recorded: a collision between two
    /// shipped keys would make one of them unreachable, and a collision
    /// against an unmeasured host would misprice it. The first is checkable
    /// and checked here; the second is a 92-in-2^32 risk the doc accepts.
    #[test]
    fn host_key_table_distinct() {
        let mut seen = FOLLOWUP_HOST_KEY.to_vec();
        seen.dedup();
        assert_eq!(seen.len(), FOLLOWUP_HOST_KEY.len(), "duplicate key");
        assert_eq!(FOLLOWUP_HOST_KEY.len(), FOLLOWUP_HOST_Q.len());
    }

    /// Clamped in the fit, asserted here, because the whole safeguard is that
    /// no single host can carry the table. 0.10 and 4.00 quantise to 6 and
    /// 255 at `q / 63.75`.
    #[test]
    fn host_scale_within_clamp() {
        for (i, q) in FOLLOWUP_HOST_Q.iter().enumerate() {
            let scale = f64::from(*q) / 63.75;
            assert!((0.09..=4.01).contains(&scale),
                    "entry {i} at {scale}");
        }
    }

    /// The hash has to be the one `fit_host_cascade.py` computes, or every
    /// key misses and the table silently does nothing. Pinned against values
    /// the generator printed, one suffix key and one exact override.
    #[test]
    fn host_scale_matches_generator() {
        let of = |s: &str| {
            let mut h = Fnv1a::new();
            h.write_cased(s.as_bytes(), true);
            h.fold()
        };
        assert_eq!(of("google-analytics.com"), 0x78ae_9fb3);
        assert_eq!(of("securepubads.g.doubleclick.net"), 0x0f02_9a53);
        // Lowercasing is part of the hash, not of the caller.
        assert_eq!(of("Google-Analytics.COM"), of("google-analytics.com"));
    }

    /// The point of suffix matching: a host never crawled, under a suffix
    /// that was, is priced by the suffix rather than defaulting to 1.0.
    #[test]
    fn host_scale_generalises_over_subdomains() {
        let ga = followup_host_scale("google-analytics.com");
        assert!(ga != 1.0, "the suffix itself should be in the table");
        // Shards that are in the sample, and ones that are not.
        for host in ["www.google-analytics.com", "region1.google-analytics.com",
                     "region99.google-analytics.com",
                     "never.seen.google-analytics.com"] {
            assert_eq!(followup_host_scale(host), ga, "{host}");
        }
        // A customer id that cannot recur is priced by its suffix too.
        let sentry = followup_host_scale("sentry.io");
        assert_eq!(followup_host_scale("o999999.ingest.sentry.io"), sentry);
    }

    /// And the limit of it: a child that disagrees keeps its own entry, which
    /// must win over the suffix.
    #[test]
    fn host_scale_exact_beats_suffix() {
        let suffix = followup_host_scale("doubleclick.net");
        let exact = followup_host_scale("ad.doubleclick.net");
        assert!(exact < suffix,
                "ad.doubleclick.net {exact} should undercut the suffix {suffix}");
        assert_eq!(followup_host_scale("securepubads.g.doubleclick.net"), suffix);
        // An unmeasured child falls through to the suffix.
        assert_eq!(followup_host_scale("unknown.doubleclick.net"), suffix);
    }

    #[test]
    fn host_scale_lookup_defaults() {
        assert_eq!(followup_host_scale("no-such-host.example"), 1.0);
        assert_eq!(followup_host_scale(""), 1.0);
        // A bare TLD is never a key, so it cannot price the whole of .com.
        assert_eq!(followup_host_scale("com"), 1.0);
        assert_eq!(followup_host_scale("net"), 1.0);
    }

    /// The table has to actually reach the cascade, and only the cascade.
    ///
    /// Compared at the level of `followup_bytes_for` rather than through
    /// `estimate_resources`, because the byte estimate is itself keyed on the
    /// URL -- two different hosts get different *direct* estimates, so a
    /// difference in the total would not show that the host scale did
    /// anything.
    #[test]
    fn host_scale_moves_the_cascade() {
        let at = |host: &str| {
            followup_bytes_for(10_000, RequestContext::SCRIPT, UrlMatch::Prefix,
                               CascadeRoot::UNKNOWN, host)
        };
        let plain = at("not-measured.example");
        assert!(at("securepubads.g.doubleclick.net") > plain);
        assert!(at("ad.doubleclick.net") < plain);
        // An unmeasured host is charged exactly the unmodulated constant.
        assert_eq!(plain, at("another-unmeasured.example"));
    }

    /// A host scale must not leak into the direct estimate, which is fitted
    /// from a log of completed requests and knows nothing about cascades.
    #[test]
    fn host_scale_does_not_touch_the_direct_estimate() {
        let url = "https://securepubads.g.doubleclick.net/tag/js/gpt.js";
        let call = |m: &str, f: bool| {
            estimate_resources(url, RequestContext::SCRIPT,
                               RequestInitiator::OTHER, m, f,
                               CascadeRoot::UNKNOWN)
                .0
        };
        assert!(call("GET", true) > call("GET", false));
        // A bodyless method still prunes nothing, table or no table.
        assert_eq!(call("HEAD", true), call("HEAD", false));
    }
}
