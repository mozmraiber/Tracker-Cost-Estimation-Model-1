"""Type stubs for the `llm_classifier_baseline` Rust extension module."""

from typing import ClassVar, final

@final
class RequestContext:
    """Kind of subresource a request is fetching.

    A PyO3 native enum, not `enum.Enum`: the variants are class attributes and
    compare equal to their integer values (`RequestContext.VIDEO == 2`).
    """

    SCRIPT: ClassVar[RequestContext]
    IMAGE: ClassVar[RequestContext]
    VIDEO: ClassVar[RequestContext]
    OTHER: ClassVar[RequestContext]

    def __int__(self) -> int: ...

def estimate_size(url: str, tracker_index: int, context: RequestContext) -> int:
    """Estimate the transfer size, in bytes, of a tracker request.

    The estimate is a conditional *mean*, so estimates over a week of browsing
    sum to an unbiased total -- which is the figure the savings dashboard
    reports. Any single request can be far off; 48% of blocked requests
    transfer no bytes at all.

    `tracker_index` is the URL's position in the Disconnect list, as
    `disconnect.tracker_index` reports it. Only URLs that list matches are in
    scope; anything else falls back to an estimate from `context` alone.
    """
