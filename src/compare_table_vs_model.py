"""The comparison `xgb-classifier/README.md` reports: the model against the table.

`llm-classifier` and `xgb-classifier` answer the same
`estimate_resources(url, context, initiator, method)` from very different
things — a compiled-in hierarchy of conditional means, and
`models/per_request/xgb_transfer_bytes.json` transpiled to Rust. This script
prices one against the other on the two axes a browser cares about, in one
table: median journey error over each dataset `tests/conftest.py` runs, and
what each estimator adds to an empty PyO3 extension.

    python src/compare_table_vs_model.py
    python src/compare_table_vs_model.py --markdown        # the README's table
    python src/compare_table_vs_model.py --dataset csv --seed 5

The numbers are the journey suite's own: this loads `conftest`'s split, samples
`conftest`'s journeys and calls `tests/solutions.py`'s predictors, so a row
here is what `tests/test_browsing_journey.py` holds to a budget — the suite
asserts those budgets rather than printing the medians, which is what this is
for. Sizes are measured the way `test_xgb_classifier_port.py` measures them,
over the empty `llm_classifier_baseline` extension.

`xgboost_shipped in Python` is the control. The crate compiles in that exact
artifact, so the two rows should agree to two decimal places; where they do
not, the interface is losing something and
`xgb-classifier/scripts/interface_ablation.py` prices what.

Anything not present is reported as `—` with a note rather than a failure: the
request logs, the trained artifacts and the built extensions are all
gitignored, and a partial table is still worth printing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# `journey`, `solutions` and `conftest` live in tests/; the model code they
# reuse lives in src/model. Same path dance as
# `xgb-classifier/scripts/interface_ablation.py`.
for _path in (ROOT / "tests", ROOT / "src" / "model"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import conftest  # noqa: E402
import http_archive  # noqa: E402
import journey  # noqa: E402
import solutions  # noqa: E402

MISSING = "—"


@dataclass(frozen=True)
class Estimator:
    """A row of the table: how to score it, and what to weigh."""

    label: str
    solution: str            # a key of `solutions.SOLUTIONS`
    package: Optional[str]   # the built extension to stat, if it is one


ESTIMATORS = (
    Estimator("llm-classifier", "llm_classifier_table", "llm_classifier"),
    Estimator("xgb-classifier", "xgb_classifier_m2cgen", "xgb_classifier"),
    # Not shipped as an extension, so it has no size: it is here to show that
    # the crate above scores what the artifact scores.
    Estimator("xgboost_shipped in Python", "xgboost_shipped", None),
)

# The empty extension both sizes are taken over. A PyO3 cdylib is ~430 KB
# before it holds anything, and what is being compared is what each estimator
# adds to one.
BASELINE_PACKAGE = "llm_classifier_baseline"

# The draws `tests/solutions.py` sets its budgets over, and the ones
# `xgb-classifier/README.md`'s table reports: the worst median of the three.
BUDGET_SEEDS = (3, 5, 7)


class Unavailable(Exception):
    """A dataset whose log is not on disk. Carries the message to print."""


def load_split(dataset: str) -> journey.Split:
    """`conftest`'s split for `dataset`, so the scores here are its scores."""
    if dataset == "csv":
        path = journey.DEFAULT_REQUEST_LOG
        if not path.exists():
            raise Unavailable(f"{path.relative_to(ROOT)} not present (gitignored); "
                              "regenerate it with the BigQuery extract in sql/")
        log = journey.load_request_log(path, page_modulus=conftest.PAGE_MODULUS)
    else:
        unavailable = http_archive.available(dataset)
        if unavailable:
            raise Unavailable(unavailable)
        log = http_archive.load_request_log(dataset)
    return journey.split_by_page(log, seed=42)


def score_dataset(dataset: str, seeds: tuple[int, ...],
                  notes: list[str]) -> dict[str, tuple[float, float]]:
    """`(median_pct, signed_median_pct)` per estimator, over one dataset.

    Over more than one seed each cell is the *worst* draw, and its signed
    median is the one from that same seed — how `tests/solutions.py` sets its
    budgets, and what the published table reports.
    """
    try:
        split = load_split(dataset)
    except Unavailable as exc:
        notes.append(f"{dataset}: {exc}")
        return {}

    actual = split.test["transfer_bytes"].clip(lower=0).to_numpy(dtype=float)
    journeys = [journey.sample_journeys(
        split.test, pages_visited=conftest.PAGES_VISITED,
        n_journeys=conftest.N_JOURNEYS, seed=seed) for seed in seeds]

    scores: dict[str, tuple[float, float]] = {}
    for estimator in ESTIMATORS:
        predictor, _budget = solutions.SOLUTIONS[estimator.solution]
        try:
            preds = predictor(split)
        except FileNotFoundError as exc:
            # What the `predict` fixture turns into a skip: an unbuilt
            # extension, or a gitignored artifact.
            notes.append(f"{dataset}: {estimator.label} not scored — {exc}")
            continue
        draws = [journey.journey_error(actual, preds, sample) for sample in journeys]
        worst = max(draws, key=lambda stats: stats["median_pct"])
        scores[estimator.label] = (worst["median_pct"], worst["signed_median_pct"])
    return scores


def shipped_sizes(notes: list[str]) -> dict[str, int]:
    """Bytes each extension adds to the empty one, for those that are built."""
    def library(package: str) -> Optional[Path]:
        # Located by path, not by import: `llm_classifier_baseline`'s
        # `#[pymodule]` is named `llm_classifier`, so importing it by its
        # package name fails. `test_xgb_classifier_port.py` stats it the same
        # way, for the same reason.
        libraries = sorted((packages / package).glob("*.abi3.so"))
        return libraries[0] if len(libraries) == 1 else None

    try:
        import llm_classifier
    except ImportError:
        notes.append("sizes omitted: llm_classifier is not built "
                     "(`maturin develop --release` in llm-classifier-python/)")
        return {}

    packages = Path(llm_classifier.__file__).parent.parent
    empty = library(BASELINE_PACKAGE)
    if empty is None:
        notes.append(f"sizes omitted: {BASELINE_PACKAGE} is not built, and it is "
                     "what the others are measured over")
        return {}

    sizes: dict[str, int] = {}
    for estimator in ESTIMATORS:
        if estimator.package is None:
            continue
        built = library(estimator.package)
        if built is None:
            notes.append(f"{estimator.label}: {estimator.package} is not built, "
                         "so it has no size here")
            continue
        sizes[estimator.label] = built.stat().st_size - empty.stat().st_size
    return sizes


def format_size(size: int) -> str:
    return f"{size / 1024:.0f} KiB" if size < 1 << 20 else f"{size / 1e6:.1f} MB"


def best(values: dict[str, float]) -> Optional[str]:
    """Which row wins a column, for the markdown table's bolding."""
    return min(values, key=values.__getitem__) if values else None


def print_plain(datasets: tuple[str, ...], seeds: tuple[int, ...],
                scores: dict[str, dict[str, tuple[float, float]]],
                sizes: dict[str, int]) -> None:
    drawn = (f"seed {seeds[0]}" if len(seeds) == 1 else
             "the worst of seeds " + ", ".join(str(seed) for seed in seeds))
    print(f"median journey error over {conftest.N_JOURNEYS} journeys of "
          f"{conftest.PAGES_VISITED} pages, {drawn}")
    print("signed in parentheses: how far the journey total runs low or high\n")

    header = f"  {'estimator':26s}" + "".join(f"{d:>22s}" for d in datasets)
    print(header + f"{'shipped size':>16s}")
    for estimator in ESTIMATORS:
        row = f"  {estimator.label:26s}"
        for dataset in datasets:
            pair = scores[dataset].get(estimator.label)
            cell = (MISSING if pair is None
                    else f"{pair[0]:.1f}% ({pair[1]:+.1f}%)")
            row += f"{cell:>22s}"
        size = sizes.get(estimator.label)
        print(row + f"{format_size(size) if size else MISSING:>16s}")


def print_markdown(datasets: tuple[str, ...],
                   scores: dict[str, dict[str, tuple[float, float]]],
                   sizes: dict[str, int]) -> None:
    """The table as `xgb-classifier/README.md` carries it, winners bolded."""
    winners = {dataset: best({label: pair[0]
                              for label, pair in scores[dataset].items()})
               for dataset in datasets}
    winners["size"] = best({label: float(size) for label, size in sizes.items()})

    # Label column, one per dataset, size column — with the datasets spanned
    # by a single "median journey error" heading, as the README writes it.
    print("| | median journey error |" + " |" * (len(datasets) - 1)
          + " shipped size |")
    print("|" + "---|" * (len(datasets) + 2))
    print("| | " + " | ".join(f"`{d}`" for d in datasets)
          + " | over an empty extension |")

    def cell(text: str, won: bool) -> str:
        return f"**{text}**" if won else text

    for estimator in ESTIMATORS:
        cells = []
        for dataset in datasets:
            pair = scores[dataset].get(estimator.label)
            cells.append(MISSING if pair is None
                         else cell(f"{pair[0]:.1f}%",
                                   winners[dataset] == estimator.label))
        size = sizes.get(estimator.label)
        cells.append(MISSING if size is None
                     else cell(format_size(size), winners["size"] == estimator.label))
        print(f"| `{estimator.label}` | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=conftest.DATASETS, action="append",
                        help="score only this dataset; repeatable "
                             "(default: every dataset the suite runs)")
    parser.add_argument("--seed", type=int, action="append",
                        help=f"journey seed, repeatable; each cell is then the "
                             f"worst draw of the seeds given, which is how the "
                             f"suite's budgets are set (default: "
                             f"{conftest.JOURNEY_SEED})")
    parser.add_argument("--worst-of-seeds", dest="seed", action="store_const",
                        const=list(BUDGET_SEEDS),
                        help=f"shorthand for --seed "
                             + " --seed ".join(str(s) for s in BUDGET_SEEDS)
                             + ", the draws the published table reports")
    parser.add_argument("--markdown", action="store_true",
                        help="emit the markdown table xgb-classifier/README.md "
                             "carries, instead of the plain one")
    args = parser.parse_args()

    datasets = tuple(dict.fromkeys(args.dataset or conftest.DATASETS))
    seeds = tuple(dict.fromkeys(args.seed or [conftest.JOURNEY_SEED]))
    notes: list[str] = []
    scores = {dataset: score_dataset(dataset, seeds, notes)
              for dataset in datasets}
    sizes = shipped_sizes(notes)

    if not any(scores.values()) and not sizes:
        raise SystemExit("nothing to compare:\n  " + "\n  ".join(notes))

    if args.markdown:
        print_markdown(datasets, scores, sizes)
    else:
        print_plain(datasets, seeds, scores, sizes)

    for index, note in enumerate(notes):
        print(f"\n{note}" if index == 0 else note)


if __name__ == "__main__":
    main()
