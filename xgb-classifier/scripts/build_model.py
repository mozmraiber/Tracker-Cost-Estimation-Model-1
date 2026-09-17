"""Compile `models/per_request/xgb_transfer_bytes.json` into the crate.

Emits three generated files, none of them checked in:

  src/trees.rs      the 370 boosting trees, transpiled to Rust by m2cgen
  src/model.bin     the TF-IDF vocabulary, the SVD basis and the two
                    target-encoding tables, as one `include_bytes!` payload
  src/generated.rs  the section sizes and scalars `lib.rs` needs to slice it

Run from anywhere:

    python xgb-classifier/scripts/build_model.py

Why a binary blob and not more generated Rust: the SVD basis alone is
50,000 x 50 float32, and as a Rust array literal that is ~120 MB of source
that rustc will not finish. `include_bytes!` puts the same bytes straight in
.rodata, and `lib.rs` reads them with `from_le_bytes`, which needs no
alignment guarantee that `include_bytes!` cannot give.

The trees are transpiled one function each rather than as m2cgen's single
`score()`. m2cgen emits the whole ensemble as one expression, which here is a
13 MB function body; rustc's time on a single function of that size is not
worth paying when the sum of 370 small ones is the same arithmetic.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from m2cgen.assemblers.boosting import XGBoostTreeModelAssembler
from m2cgen.interpreters import PythonInterpreter, RustInterpreter
from m2cgen.interpreters.python.code_generator import PythonCodeGenerator
from m2cgen.interpreters.rust.code_generator import RustCodeGenerator

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
# `journey` (the request log and its split) and `url_embeddings` (the class the
# shipped embedder was pickled from) both live outside this crate.
for path in (ROOT / "tests", ROOT / "src" / "model"):
    sys.path.insert(0, str(path))

MODEL_DIR = ROOT / "models" / "per_request"
SHIPPED_MODEL = MODEL_DIR / "xgb_transfer_bytes.json"
SHIPPED_EMBEDDER = MODEL_DIR / "url_embedder.joblib"

MAGIC = b"XGBCLS01"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    """FNV-1a, matching `Fnv1a` in lib.rs. Not folded: a 64-bit key makes a
    collision over 50,000 vocabulary terms a ~1e-10 event, where the 32 bits
    `llm-classifier` folds to would expect one."""
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Trees
# --------------------------------------------------------------------------- #

class ExactFloat32Literals:
    """Print every literal at full `f64` precision.

    m2cgen holds the artifact's split conditions and leaves as `np.float32` --
    which they are -- and prints them with float32's shortest round-tripping
    decimal, so `0.8973824`. Parsed back as an `f64` that is *not* the float32
    the booster compares against: it is up to half a float32 ulp away.

    XGBoost was fitted with `tree_method="hist"`, whose split conditions are
    observed feature values rather than midpoints between them, so a feature
    landing exactly on a threshold is the common case rather than a corner
    one -- and on exactly those the shifted literal sends the request down the
    wrong branch. Left alone it moved 22% of the CSV extract's estimates and
    7% of its summed bytes.

    `float(np.float32(x))` is the exact value of the float32, and an `f64`
    comparison between two exactly-representable float32s agrees with the
    float32 comparison, which is the one the booster makes.
    """

    def num_value(self, value):
        return super().num_value(float(value))


class SliceRustCodeGenerator(ExactFloat32Literals, RustCodeGenerator):
    """m2cgen's Rust generator, emitting `&[f64]` for the feature argument.

    It declares it `Vec<f64>` otherwise, which would take ownership and so
    force a clone per tree. Trees declare no vector locals, so overriding the
    one type is enough.
    """

    vector_type = "&[f64]"


class ExactPythonCodeGenerator(ExactFloat32Literals, PythonCodeGenerator):
    """m2cgen's Python generator, so `--check` verifies the same thresholds."""


class SliceRustInterpreter(RustInterpreter):
    def __init__(self, indent: int = 4, function_name: str = "score",
                 *args, **kwargs) -> None:
        super().__init__(indent=indent, function_name=function_name,
                         *args, **kwargs)
        self._cg = SliceRustCodeGenerator(indent=indent)


class ExactPythonInterpreter(PythonInterpreter):
    def __init__(self, indent: int = 4, function_name: str = "score",
                 *args, **kwargs) -> None:
        super().__init__(indent=indent, function_name=function_name,
                         *args, **kwargs)
        self._cg = ExactPythonCodeGenerator(indent=indent)


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(SHIPPED_MODEL))
    return model


def tree_count(model: xgb.XGBRegressor) -> int:
    """How many trees `model.predict` actually applies.

    The artifact stores 390 but was early-stopped, and the sklearn wrapper
    predicts with `best_iteration + 1` of them. m2cgen reads the now-removed
    `best_ntree_limit` attribute to learn this and so reads nothing, which is
    why the limit is applied here instead.
    """
    best = getattr(model, "best_iteration", None)
    return model.get_booster().num_boosted_rounds() if best is None else best + 1


def base_score(model: xgb.XGBRegressor) -> float:
    """The intercept, in the *original* prediction space.

    XGBoost >= 1.7 stores `base_score` un-linked, so for this model's
    `reg:tweedie` (a log link) the margin intercept is `ln(base_score)` and the
    prediction is `base_score * exp(sum_of_leaves)`. m2cgen adds `base_score`
    into the margin directly, which for a log-link objective is wrong by
    orders of magnitude -- so the trees are transpiled with no intercept at all
    and lib.rs multiplies by this. `--check` verifies that against
    `model.predict`.
    """
    return float(np.asarray(model.get_params()["base_score"]).reshape(-1)[0])


def emit_trees(model: xgb.XGBRegressor, n_trees: int) -> str:
    assembler = XGBoostTreeModelAssembler(model)
    trees = assembler._all_estimator_params[:n_trees]

    parts = [
        "// @generated by scripts/build_model.py -- do not edit by hand.",
        f"// Source: {SHIPPED_MODEL.relative_to(ROOT)},"
        f" trees 0..{n_trees} of {len(assembler._all_estimator_params)}.",
        "//",
        "// One function per boosting tree, transpiled from the artifact's JSON",
        "// dump by m2cgen. `input` is the 80-element feature vector",
        "// `features::build` returns, in the order the booster names them.",
        "//",
        "// The comparisons carry XGBoost's missing-value handling: m2cgen",
        "// orients each test so that the node's `missing` child sits on the",
        "// `else` branch, which is where a NaN feature lands. `features::build`",
        "// passes no NaNs -- the features it cannot observe are zero, which is",
        "// what the training log holds for them -- but the routing is faithful",
        "// if a future caller does.",
        "",
        "#![allow(clippy::all)]",
        "",
    ]
    for i, tree in enumerate(trees):
        ast = assembler._assemble_tree(tree)
        parts.append(SliceRustInterpreter(function_name=f"t{i}").interpret(ast))

    parts += [
        "",
        "/// Sum of every tree's leaf, i.e. the boosting margin with no",
        "/// intercept. `lib.rs` turns it into bytes.",
        "pub fn margin(input: &[f64]) -> f64 {",
        "    0.0",
    ]
    parts += [f"        + t{i}(input)" for i in range(len(trees))]
    parts += [
        "}",
        "",
        "// This file and generated.rs are written by the same run, and a",
        "// mismatched pair would silently sum the wrong number of trees.",
        f"const _: () = assert!(crate::generated::N_TREES == {len(trees)});",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #

def load_embedder():
    import joblib
    pipeline = joblib.load(SHIPPED_EMBEDDER)
    return pipeline["tfidf"], pipeline["svd"]


def target_encodings() -> tuple[dict[bytes, float], dict[bytes, float], float]:
    """The three target encodings `engineer_features` derives from the log.

    Keyed and computed exactly as it does -- medians of `transfer_bytes` over
    the CSV extract's *train* half, which is the half the shipped artifact was
    fitted on. Read `solutions.csv_training_split` for why that frame and not
    the evaluation dataset's own.
    """
    from solutions import csv_training_split

    train = csv_training_split().train
    domain = train.groupby("tracker_domain")["transfer_bytes"].median()
    domain_type = train.groupby(
        ["tracker_domain", "resource_type"])["transfer_bytes"].median()

    domains = {fnv1a64(str(k).encode()): float(v) for k, v in domain.items()}
    domain_types = {
        fnv1a64(f"{d}\0{r}".encode()): float(v)
        for (d, r), v in domain_type.items()
    }
    return domains, domain_types, float(train["transfer_bytes"].median())


def sorted_pairs(table: dict[int, float]) -> tuple[list[int], list[float]]:
    keys = sorted(table)
    return keys, [table[k] for k in keys]


def emit_blob(tfidf, svd, domains, domain_types) -> tuple[bytes, dict[str, int]]:
    """Pack the vocabulary, the SVD basis and the encoding tables.

    Sections in a fixed order so `generated.rs` can give `lib.rs` const
    offsets. Everything is little-endian and read back with `from_le_bytes`,
    so no section needs padding to its natural alignment.

    `term_hash` is sorted for binary search and carries the term's *vocabulary*
    index alongside, rather than being reordered into hash order: `idf` and the
    basis stay in vocabulary order, which lets lib.rs accumulate a URL's terms
    in the order scipy's sparse matmul does. Float addition is not associative,
    and the ensemble is sensitive enough to a 1e-7 shift in an embedding that
    the order is worth keeping.
    """
    names = tfidf.get_feature_names_out()
    vocab = tfidf.vocabulary_
    n_terms = len(names)
    components = svd.components_          # (50, n_terms), float32 as stored
    n_components = components.shape[0]
    assert components.shape[1] == n_terms

    term_hashes = {}
    for name in names:
        h = fnv1a64(str(name).encode())
        assert h not in term_hashes, f"64-bit hash collision on {name!r}"
        term_hashes[h] = int(vocab[name])

    ordered = sorted(term_hashes)
    domain_keys, domain_vals = sorted_pairs(domains)
    dt_keys, dt_vals = sorted_pairs(domain_types)

    sections = [
        struct.pack(f"<{n_terms}Q", *ordered),
        struct.pack(f"<{n_terms}I", *(term_hashes[h] for h in ordered)),
        tfidf.idf_.astype("<f4").tobytes(),
        # Term-major: a request touches a few dozen terms and reads all 50
        # basis entries of each, so the 50 belong next to one another.
        np.ascontiguousarray(components.T, dtype="<f4").tobytes(),
        struct.pack(f"<{len(domain_keys)}Q", *domain_keys),
        np.asarray(domain_vals, dtype="<f4").tobytes(),
        struct.pack(f"<{len(dt_keys)}Q", *dt_keys),
        np.asarray(dt_vals, dtype="<f4").tobytes(),
    ]
    return MAGIC + b"".join(sections), {
        "n_terms": n_terms,
        "n_components": n_components,
        "n_domains": len(domain_keys),
        "n_domain_types": len(dt_keys),
    }


def emit_generated_rs(counts: dict[str, int], global_median: float,
                      bs: float, n_trees: int) -> str:
    return f"""// @generated by scripts/build_model.py -- do not edit by hand.
//
// Shapes and scalars for the `model.bin` payload `lib.rs` slices. Every
// section offset is a const expression over these, so nothing about the
// blob's layout is discovered at run time.

/// Vocabulary terms the shipped TF-IDF kept: unigrams and bigrams of the
/// URL-path tokens, capped at `max_features`.
pub const N_TERMS: usize = {counts['n_terms']};

/// SVD output dimensionality, i.e. how many `url_emb_*` features there are.
pub const N_COMPONENTS: usize = {counts['n_components']};

/// Tracker domains the training half of the log carried.
pub const N_DOMAINS: usize = {counts['n_domains']};

/// (tracker domain, resource type) pairs it carried.
pub const N_DOMAIN_TYPES: usize = {counts['n_domain_types']};

/// Median `transfer_bytes` over the whole training half, which is what both
/// target encodings fall back to for a domain never seen.
pub const GLOBAL_MEDIAN: f64 = {global_median!r};

/// The artifact's intercept, in the original prediction space -- see
/// `base_score` in scripts/build_model.py.
pub const BASE_SCORE: f64 = {bs!r};

/// Trees `margin` sums. Fewer than the artifact stores, because it was
/// early-stopped; see `tree_count` in scripts/build_model.py.
pub const N_TREES: usize = {n_trees};
"""


# --------------------------------------------------------------------------- #

def check(model: xgb.XGBRegressor, n_trees: int, bs: float) -> None:
    """Assert `base_score * exp(margin)` reproduces `model.predict`.

    Uses m2cgen's Python backend rather than the Rust one, so what is verified
    is the tree transpilation and the intercept reconstruction -- not the Rust
    build, which tests/ covers end to end.
    """
    import importlib.util
    import tempfile

    assembler = XGBoostTreeModelAssembler(model)
    trees = assembler._all_estimator_params[:n_trees]
    src = "\n".join(
        ExactPythonInterpreter(function_name=f"t{i}").interpret(
            assembler._assemble_tree(t))
        for i, t in enumerate(trees))
    src += ("\ndef margin(x):\n    return "
            + " + ".join(f"t{i}(x)" for i in range(len(trees))) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trees.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location("trees", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rng = np.random.default_rng(7)
        rows = 40
        features = rng.normal(size=(rows, 80))
        features[:, :2] = np.abs(rng.normal(3000, 4000, size=(rows, 2)))
        features[:, 2:5] = rng.integers(0, 9, size=(rows, 3))
        features[:, 5:30] = rng.integers(0, 2, size=(rows, 25))
        features[:, 20:24] = np.nan     # the two this crate cannot observe

        want = model.predict(features.astype(np.float32))
        got = np.array([bs * math.exp(module.margin(list(row)))
                        for row in features])
        worst = float(np.max(np.abs(got - want) / np.maximum(want, 1e-12)))
        assert worst < 1e-4, f"transpiled trees disagree by {worst:.2e}"
        print(f"  checked 40 rows against model.predict: worst {worst:.1e}"
              " relative (float32 leaves accumulated in float64)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the transpiled trees against model.predict")
    args = parser.parse_args()

    for artifact in (SHIPPED_MODEL, SHIPPED_EMBEDDER):
        if not artifact.exists():
            raise SystemExit(
                f"{artifact.relative_to(ROOT)} is missing (it is gitignored); "
                "retrain it with src/model/train_multi_target.py")

    started = time.time()
    model = load_model()
    n_trees, bs = tree_count(model), base_score(model)
    print(f"model: {n_trees} trees, base_score {bs:.3f}, "
          f"{len(model.get_booster().feature_names)} features")

    if args.check:
        check(model, n_trees, bs)

    trees_rs = emit_trees(model, n_trees)
    (CRATE / "src" / "trees.rs").write_text(trees_rs)
    print(f"  src/trees.rs      {len(trees_rs) / 1e6:6.1f} MB")

    tfidf, svd = load_embedder()
    domains, domain_types, global_median = target_encodings()
    blob, counts = emit_blob(tfidf, svd, domains, domain_types)
    (CRATE / "src" / "model.bin").write_bytes(blob)
    print(f"  src/model.bin     {len(blob) / 1e6:6.1f} MB "
          f"({counts['n_terms']} terms x {counts['n_components']} components, "
          f"{counts['n_domains']} domains, "
          f"{counts['n_domain_types']} domain+type pairs)")

    generated = emit_generated_rs(counts, global_median, bs, n_trees)
    (CRATE / "src" / "generated.rs").write_text(generated)
    print(f"  src/generated.rs  {len(generated) / 1e3:6.1f} kB")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
