#!/usr/bin/env python
"""Benchmark rustling.wordseg vs pure Python wordseg.

Compares word segmentation speed using HKCanCor corpus data.

Usage:
    python benchmarks/run_wordseg.py
    python benchmarks/run_wordseg.py --quick
    python benchmarks/run_wordseg.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.data import load_hkcancor, wordseg_data  # noqa: E402


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    implementation: str
    num_sentences: int
    time_seconds: float
    iterations: int
    sentences_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        total_sentences = self.num_sentences * self.iterations
        self.sentences_per_second = total_sentences / self.time_seconds


def time_function(
    func: Callable[[], Any],
    iterations: int = 5,
    warmup: int = 1,
) -> float:
    """Time a function over multiple iterations.

    Returns
    -------
    float
        Total time in seconds.
    """
    for _ in range(warmup):
        func()

    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return sum(times)


def benchmark_longest_string_matching(
    rustling_cls: type | None,
    wordseg_cls: type | None,
    training_data: list[tuple[str, ...]],
    test_sentences: list[str],
    max_word_length: int,
    iterations: int = 5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark LongestStringMatching implementations."""
    results = []

    for name, cls in [("rustling", rustling_cls), ("wordseg", wordseg_cls)]:
        if cls is None:
            results.append(None)
            continue

        model = cls(max_word_length=max_word_length)
        model.fit(training_data)

        def predict() -> None:
            result = model.predict(test_sentences)
            # Force evaluation of lazy iterators (wordseg returns map objects)
            list(result)

        total_time = time_function(predict, iterations=iterations)

        results.append(
            BenchmarkResult(
                name="LongestStringMatching",
                implementation=name,
                num_sentences=len(test_sentences),
                time_seconds=total_time,
                iterations=iterations,
            )
        )

    return results[0], results[1]


def print_result(result: BenchmarkResult) -> None:
    """Print a benchmark result."""
    print(f"  {result.implementation}:")
    print(
        f"    Total time: {result.time_seconds:.4f}s ({result.iterations} iterations)"
    )
    print(f"    Sentences/second: {result.sentences_per_second:,.0f}")


def print_comparison(
    rustling_result: BenchmarkResult, wordseg_result: BenchmarkResult
) -> None:
    """Print comparison between rustling and wordseg."""
    if rustling_result.time_seconds > 0:
        speedup = wordseg_result.time_seconds / rustling_result.time_seconds
    else:
        speedup = float("inf")
    print(f"\n  ⚡ Speedup: {speedup:.1f}x faster")


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks.

    Parameters
    ----------
    quick : bool, default=False
        If True, use a smaller data subset.
    verbose : bool, default=True
        If True, print results.

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    # Try to import both implementations
    rustling_lsm = None
    wordseg_lsm = None

    try:
        from rustling.wordseg import LongestStringMatching as RustlingLSM

        rustling_lsm = RustlingLSM
        if verbose:
            print("✓ rustling.wordseg loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"✗ rustling.wordseg not available: {e}")

    try:
        from wordseg import LongestStringMatching as WordsegLSM

        wordseg_lsm = WordsegLSM
        if verbose:
            print("✓ wordseg (pure Python) loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"✗ wordseg (pure Python) not available: {e}")

    if rustling_lsm is None and wordseg_lsm is None:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    # Load data
    if verbose:
        print("\nLoading HKCanCor corpus...")
    tagged_sents = load_hkcancor()
    training_data, test_sentences = wordseg_data(tagged_sents)

    if quick:
        training_data = training_data[:500]
        test_sentences = test_sentences[:100]
        iterations = 3
    else:
        iterations = 5

    # Compute max_word_length from training data
    max_word_length = max(len(word) for sent in training_data for word in sent)

    if verbose:
        print(f"Training sentences: {len(training_data)}")
        print(f"Test sentences: {len(test_sentences)}")
        print(f"Max word length: {max_word_length}")

    all_results: dict[str, Any] = {"benchmarks": {}}

    print("\n" + "=" * 60)
    print("WORDSEG BENCHMARK: Rustling (Rust) vs wordseg (Python)")
    print("=" * 60)

    # Benchmark LongestStringMatching
    if verbose:
        print("\n📊 LongestStringMatching:")

    lsm_rustling, lsm_wordseg = benchmark_longest_string_matching(
        rustling_lsm,
        wordseg_lsm,
        training_data,
        test_sentences,
        max_word_length=max_word_length,
        iterations=iterations,
    )

    if lsm_rustling and verbose:
        print_result(lsm_rustling)
    if lsm_wordseg and verbose:
        print_result(lsm_wordseg)
    if lsm_rustling and lsm_wordseg and verbose:
        print_comparison(lsm_rustling, lsm_wordseg)

    all_results["benchmarks"]["LongestStringMatching"] = {
        "rustling": lsm_rustling.__dict__ if lsm_rustling else None,
        "wordseg": lsm_wordseg.__dict__ if lsm_wordseg else None,
    }

    # Compute speedups
    speedups: dict[str, float] = {}
    for algo in ["LongestStringMatching"]:
        bench = all_results["benchmarks"].get(algo, {})
        r = bench.get("rustling")
        w = bench.get("wordseg")
        if r and w and r["time_seconds"] > 0:
            speedups[algo] = w["time_seconds"] / r["time_seconds"]
    all_results["speedups"] = speedups

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.wordseg vs pure Python wordseg"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller data",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (useful with --export)",
    )

    args = parser.parse_args()

    results = run_benchmarks(
        quick=args.quick,
        verbose=not args.quiet,
    )

    if args.export:
        export_path = Path(args.export)
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
