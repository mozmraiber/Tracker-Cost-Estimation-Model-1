"""Type stubs for the `llm_classifier` Rust extension module."""

from typing import ClassVar, final

@final
class RequestInitiator:
    """What caused the browser to issue the request.

    A PyO3 native enum, not `enum.Enum`: the variants are class attributes and
    compare equal to their integer values.

    Identical to `xgb_classifier.RequestInitiator`, discriminants included, so
    the two extensions stay swappable. The request log's `preflight`, `FedCM`
    and `preload` have no variant of their own and belong in `UNKNOWN`.

    `estimate_resources` takes this and does not use it. The table is not keyed
    on the initiator because the URL path already carries what it would say --
    script-initiated GETs average 17.2 KB and parser-discovered ones 3.5 KB,
    and the table separates them to within 6% without being told which is
    which. It is part of the interface because both extensions share one, and
    because `xgb_classifier`, which has no path-level table, gets 19 points of
    journey error out of it. `llm-classifier/scripts/build_table.py` records
    the measurement under `INITIATOR_UNUSED`.
    """

    PARSER: ClassVar[RequestInitiator]
    SCRIPT: ClassVar[RequestInitiator]
    OTHER: ClassVar[RequestInitiator]
    UNKNOWN: ClassVar[RequestInitiator]

    def __int__(self) -> int: ...

@final
class RequestContext:
    """Kind of subresource a request is fetching.

    A PyO3 native enum, not `enum.Enum`: the variants are class attributes and
    compare equal to their integer values (`RequestContext.VIDEO == 2`).

    `OTHER` is the catch-all for resource types with no variant of their own.
    Some variants may have had no training rows -- their estimate then comes
    from the mean over every request rather than from their own.
    """

    SCRIPT: ClassVar[RequestContext]
    IMAGE: ClassVar[RequestContext]
    VIDEO: ClassVar[RequestContext]
    OTHER: ClassVar[RequestContext]
    AUDIO: ClassVar[RequestContext]
    CSS: ClassVar[RequestContext]
    FONT: ClassVar[RequestContext]
    HTML: ClassVar[RequestContext]
    TEXT: ClassVar[RequestContext]
    WASM: ClassVar[RequestContext]
    XML: ClassVar[RequestContext]

    def __int__(self) -> int: ...

@final
class CascadeRoot:
    """Whether this page load already had a cascading tracker on this host.

    A PyO3 native enum, not `enum.Enum`: the variants are class attributes and
    compare equal to their integer values.

    Read by `estimate_resources` only when `include_followups` is true. Two
    blocked requests to the same tracker host on one page do not prune two
    subtrees -- they prune one, twice -- so `REPEAT` is charged no cascade at
    all and `FIRST` is charged more than the flat per-request figure.

    `UNKNOWN` is the default and is not a hedge: it charges the crawl average
    over both cases, which is exactly what `estimate_resources` returned
    before this argument existed. A browser knows which hosts it has already
    blocked on a page; a script scoring a request log row by row does not, and
    should say so rather than guess.

    Unlike the other two enums here this one has no `xgb_classifier`
    counterpart, which is why it is defaulted -- the two modules stay
    swappable.
    """

    FIRST: ClassVar[CascadeRoot]
    REPEAT: ClassVar[CascadeRoot]
    UNKNOWN: ClassVar[CascadeRoot]

    def __int__(self) -> int: ...

def estimate_resources(
    url: str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: str,
    include_followups: bool = False,
    cascade_root: CascadeRoot = ...,
) -> tuple[int, float]:
    """Estimate the transfer size and CPU time of a tracker request.

    Returns `(bytes, cpu_ms)`. A plain tuple rather than a named struct: a
    pyo3 `#[pyclass]` costs ~18 KiB of type-object machinery, and this
    extension is held to a size budget by `tests/test_browsing_journey.py`.

    The estimates are conditional *means*, so they sum to an unbiased total
    over a week of browsing -- which is the figure the savings dashboard
    reports. Any single request can be far off; 48% of blocked requests
    transfer no bytes at all.

    The first four arguments are required, which is the one way this differs
    from `xgb_classifier.estimate_resources`, whose last two default. A browser
    knows all four when it blocks a request. Pass `""` for a method that
    really is not known: it reads the table exactly as a GET does, which is
    what this function answered before it took a method at all.

    `include_followups` asks a different question rather than supplying
    another fact about the request. False -- the default, and what this
    function answered before the argument existed -- prices the blocked
    request: the bytes ETP refused and the CPU they would have cost. True also
    prices the requests that request would have issued and now never will,
    because blocking a tag manager or an ad loader prunes the subtree beneath
    it; on the paired top-500 crawl that takes the estimated total from
    25.0 MB to 58.3 MB a pass, against 55.9 MB measured over
    Disconnect-matched requests.
    Sum with it true when the claim is what the network and the CPU were
    spared, which is what a savings dashboard claims; leave it false when the
    estimate is being compared against an observation of one request.

    It is much the rougher of the two. The direct estimate is fitted per URL
    over a log of real blocked requests; the cascade is one constant per
    context, charged per blocked request rather than per byte -- a blocked
    script is priced at the subtree an average one prunes, whatever its own
    size, because fitting both terms against the crawl leaves the size term
    near zero. The constant is a lower bound rather than a measurement: it is
    47 KB, the part of a blocked tracker's subtree that lands back on hosts
    the Disconnect list names, which the crawl measures three ways inside
    six percent and HTTP Archive's request tree bounds above at 62 KB per
    script root. The rest of the subtree -- ad creatives and iframes on
    hosts no list names -- is real and is not in it.

    One thing refines it, and it is the only one that survived: whether the
    page has already had a tracker blocked on the same host. Pass
    `cascade_root` and a first block is charged 61 KB while a repeat is
    charged nothing, which beats the flat per-request figure on 99% of
    bootstrap resamples of the crawl. Nothing else does: not the page a
    request is on, not the tracker serving it, not the request's own size,
    not how many were blocked on the page, and not how specifically the table
    recognised the URL (`classify_url`) -- all were measured and none
    separated.
    Read a cascade-inclusive total as good to a factor of two, and see
    `FOLLOWUP_BYTES_PER_REQUEST` in `llm-classifier/src/lib.rs` for the
    evidence and `llm-classifier/scripts/fit_followups.py` to re-run it.

    `method` is used for one thing, which is the only thing the table found it
    worth using it for: a method whose response carries no body -- HEAD, or the
    OPTIONS of a CORS preflight -- is answered from the method alone, at the
    9.1 bytes such requests average, rather than from the URL. The URL prices a
    preflight as the POST whose path it shares, at 1,262 bytes, which is 84x
    further out in per-request error over the 1.37% of blocked requests
    concerned. It is a fix to individual answers rather than to a total, since
    those requests carry 0.0017% of all blocked bytes. POST is not in that
    category: its responses are mostly empty too, but a beacon endpoint has a
    path of its own and the table reads it better than the method does.

    `initiator` is accepted and unused -- see `RequestInitiator`.

    Otherwise keyed on the URL's path and `context`. The table is fitted over
    the requests the Disconnect list matches, since those are the ones ETP
    blocks, but the caller does not have to resolve the tracker: a URL from
    outside that population still answers, from whatever level of the hierarchy
    its path reaches.

    `cpu_ms` is PROVISIONAL and order-of-magnitude: it is `bytes` scaled by a
    per-context ms/KiB rate plus a fixed per-request term, not a fitted
    quantity. Independent evidence puts the blended rate for blocked trackers
    somewhere between 1.3 and 4.0 ms/KiB. Prefer `bytes` where only one number
    can be relied on.
    """


def classify_url(
    url: str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: str,
) -> str:
    """How specifically the size table recognises `url`.

    One of `"path"`, `"template"`, `"prefix"`, `"host"`, `"host_template"`,
    `"ext_query"`, `"ext"`, `"context"` or `"bodyless"`, most specific first.
    A string rather than an enum for the reason `estimate_resources` returns
    a plain tuple: a `#[pyclass]` costs ~18 KiB against this extension's size
    budget.

    The estimator answers from a hierarchy of fallbacks -- the exact path,
    then the path with versions collapsed, the asset family, the host, the
    host with generated labels collapsed, extension-and-query-length, and
    extension alone -- and stops at the first rung with an entry. This
    reports which rung that was, from the same walk that produces the
    estimate, so the two cannot disagree. `"context"` means no rung matched
    and the answer is the mean over a whole request context; `"bodyless"`
    means the method forbids a response body, so the URL was never consulted.

    It classifies the *URL*, not the response, which is what makes it usable
    where the estimate is: at block time, from what the browser already has.
    Take it as the estimator's own statement of how much it knew before it
    answered -- `"path"` is a seen asset, `"ext"` is the mean of every `.js`
    in HTTP Archive.

    What it is for is grading. `tests/top500.py` cuts its byte bounds by this
    and finds that the rungs are not equally calibrated: on the paired
    top-500 crawl `prefix` runs +13.8% and `ext_query` -24.3%, which is
    ±5 points of the crawl's total cancelling inside an aggregate that reads
    +2.1%. The same cut does *not* refine the cascade -- see
    `estimate_resources` and `llm-classifier/scripts/fit_followups.py`.

    Takes the same four arguments as `estimate_resources` and for the same
    reasons: `context` is part of every lookup key, so one URL can match at
    different rungs as a script and as an image, and a bodyless `method`
    skips the walk. `initiator` is accepted and unused.
    """
