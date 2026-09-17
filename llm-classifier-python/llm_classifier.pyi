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

def estimate_resources(
    url: str,
    context: RequestContext,
    initiator: RequestInitiator,
    method: str,
) -> tuple[int, float]:
    """Estimate the transfer size and CPU time of a tracker request.

    Returns `(bytes, cpu_ms)`. A plain tuple rather than a named struct: a
    pyo3 `#[pyclass]` costs ~18 KiB of type-object machinery, and this
    extension is held to a size budget by `tests/test_browsing_journey.py`.

    The estimates are conditional *means*, so they sum to an unbiased total
    over a week of browsing -- which is the figure the savings dashboard
    reports. Any single request can be far off; 48% of blocked requests
    transfer no bytes at all.

    All four arguments are required, which is the one way this differs from
    `xgb_classifier.estimate_resources`, whose last two default. A browser
    knows all four when it blocks a request. Pass `""` for a method that
    really is not known: it reads the table exactly as a GET does, which is
    what this function answered before it took a method at all.

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
