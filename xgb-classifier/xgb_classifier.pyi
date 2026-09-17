"""Type stubs for the `xgb_classifier` Rust extension module."""

from typing import ClassVar, final

@final
class RequestInitiator:
    """What caused the browser to issue the request.

    A PyO3 native enum, not `enum.Enum`: the variants are class attributes and
    compare equal to their integer values.

    The artifact's `init_script`, `init_parser` and `init_other` one-hots, as
    one value. Only those three are in the feature matrix, so the log's
    `preflight`, `FedCM` and `preload` have no variant and belong in `UNKNOWN`,
    which is what `engineer_features` gives them too: all three one-hots zero.

    This is the single most valuable thing the estimator can be told beyond the
    URL. Script-initiated tracker requests average 12.2 KB and account for
    87.7% of all blocked bytes; parser-discovered ones average 2.2 KB.
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

    `llm_classifier.RequestContext`'s eleven variants with their discriminants,
    so a caller of either can pass the same value, plus `JSON` appended. The
    shared discriminants keep the values they had because that crate's
    generated table hashed them into every lookup key. The variant selects the
    request log's `resource_type`, which keys the booster's `rt_*` one-hots and
    its `domain_type_median` target encoding.

    `OTHER` is the catch-all for a resource type the log does not name. The
    mapping is otherwise one-to-one with the log's own values, so no two of
    them share a row of the model's target encodings -- which is why `JSON` is
    here and `llm_classifier` has no counterpart for it: its table was fitted
    with `json` and `other` binned together.
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
    JSON: ClassVar[RequestContext]

    def __int__(self) -> int: ...

def estimate_resources(
    url: str,
    context: RequestContext,
    initiator: RequestInitiator = ...,
    method: str | None = None,
) -> tuple[int, float]:
    """Estimate the transfer size and CPU time of a tracker request.

    Returns `(bytes, cpu_ms)`, the same shape `llm_classifier` returns.

    `initiator` and `method` are optional and default to unknown, which leaves
    the four features derived from them zero — what `engineer_features`
    produces for a log row recording neither. So a two-argument call works and
    matches `llm_classifier`'s signature, whose table has no use for either.
    Supplying them is worth a lot: on the CSV extract it is the difference
    between 28.7% and 8.2% median journey error.

    `bytes` comes from `models/per_request/xgb_transfer_bytes.json`, the
    `xgboost_shipped` artifact, compiled into the extension by
    `scripts/build_model.py`: its 370 trees transpiled to Rust by m2cgen, plus
    a port of the `engineer_features` pipeline that feeds them -- the two
    target encodings, the URL-structure and regex features, and the TF-IDF and
    truncated-SVD embedding of the URL path.

    The estimate is a conditional mean, so it is meant to sum to an unbiased
    weekly total. On the CSV extract it roughly is, at -6.5%; on the HTTP
    Archive export it is not, at -30%, which is the transfer failure
    `tests/solutions.py` records for `xgboost_shipped` in
    `KNOWN_UNCALIBRATED`. They are the same numbers because, given every
    argument, this is the same predictor: `engineer_features` reads nothing
    this interface cannot carry.

    `cpu_ms` is PROVISIONAL and order-of-magnitude, and is `llm_classifier`'s
    calculation unchanged: `bytes` scaled by a per-context ms/KiB rate plus a
    fixed per-request term, not a fitted quantity. Prefer `bytes` where only
    one number can be relied on.
    """

def feature_vector(
    url: str,
    context: RequestContext,
    initiator: RequestInitiator = ...,
    method: str | None = None,
) -> list[float]:
    """The 80 features `estimate_resources` hands the booster.

    Not part of the estimator's interface -- `llm_classifier` has no
    counterpart and callers have no use for it. It exists so that the crate's
    correctness claim can be tested directly: given the inputs the interface
    can carry, these are `engineer_features`' own columns, in the order the
    artifact names them. `tests/test_xgb_classifier_port.py` asserts them equal
    column by column, which a prediction comparison could not do -- the
    ensemble is sensitive enough that one ulp in one embedding changes the
    answer, so a mismatched prediction says nothing about which feature is
    wrong.
    """
