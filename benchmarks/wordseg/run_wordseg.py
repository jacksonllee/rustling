#!/usr/bin/env python
"""Benchmark rustling.wordseg vs pure Python wordseg.

This script compares the performance of rustling's Rust-based word segmentation
implementations against the pure Python wordseg package.

Usage:
    python benchmarks/run_wordseg.py
    python benchmarks/run_wordseg.py --quick      # Quick sanity check
    python benchmarks/run_wordseg.py --full       # Full benchmark suite
    python benchmarks/run_wordseg.py --export     # Export results to JSON
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    implementation: str
    num_sentences: int
    avg_sentence_length: int
    time_seconds: float
    iterations: int
    sentences_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        total_sentences = self.num_sentences * self.iterations
        self.sentences_per_second = total_sentences / self.time_seconds


@dataclass
class ComparisonResult:
    """Comparison between rustling and wordseg."""

    benchmark_name: str
    rustling_time: float
    wordseg_time: float
    speedup: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rustling_time > 0:
            self.speedup = self.wordseg_time / self.rustling_time
        else:
            self.speedup = float("inf")


def generate_training_data(
    num_words: int = 1000,
    min_word_len: int = 2,
    max_word_len: int = 10,
    vocab_chars: str = string.ascii_lowercase,
) -> list[tuple[str, ...]]:
    """Generate random training sentences.

    Args:
        num_words: Total number of unique words to generate.
        min_word_len: Minimum word length.
        max_word_len: Maximum word length.
        vocab_chars: Characters to use for generating words.

    Returns:
        A list of sentences (tuples of words).
    """
    # Generate vocabulary
    vocab = set()
    while len(vocab) < num_words:
        length = random.randint(min_word_len, max_word_len)
        word = "".join(random.choices(vocab_chars, k=length))
        vocab.add(word)
    vocab_list = list(vocab)

    # Generate sentences
    sentences = []
    words_per_sent = 5
    num_sents = num_words // words_per_sent
    for _ in range(num_sents):
        sent = tuple(random.choices(vocab_list, k=words_per_sent))
        sentences.append(sent)

    return sentences


def generate_test_sentences(
    training_data: list[tuple[str, ...]],
    num_sentences: int = 100,
    words_per_sentence: int = 10,
) -> list[str]:
    """Generate unsegmented test sentences from training vocabulary.

    Args:
        training_data: Training data to extract vocabulary from.
        num_sentences: Number of test sentences to generate.
        words_per_sentence: Words per sentence.

    Returns:
        List of unsegmented sentence strings.
    """
    # Extract vocabulary from training data
    vocab = []
    for sent in training_data:
        vocab.extend(sent)
    vocab = list(set(vocab))

    if not vocab:
        # Fallback if no training data
        vocab = ["hello", "world", "test", "word"]

    sentences = []
    for _ in range(num_sentences):
        words = random.choices(vocab, k=words_per_sentence)
        sentences.append("".join(words))

    return sentences


def time_function(
    func: Callable[[], Any],
    iterations: int = 5,
    warmup: int = 1,
) -> tuple[float, list[float]]:
    """Time a function over multiple iterations.

    Args:
        func: Function to time.
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).

    Returns:
        Tuple of (total_time, list_of_times).
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Force garbage collection before timing
    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times), times


def benchmark_longest_string_matching(
    rustling_cls: type | None,
    wordseg_cls: type | None,
    training_data: list[tuple[str, ...]],
    test_sentences: list[str],
    max_word_length: int = 10,
    iterations: int = 5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark LongestStringMatching implementations.

    Args:
        rustling_cls: rustling.wordseg.LongestStringMatching class.
        wordseg_cls: wordseg.LongestStringMatching class.
        training_data: Training sentences.
        test_sentences: Test sentences to segment.
        max_word_length: Maximum word length parameter.
        iterations: Number of iterations for timing.

    Returns:
        Tuple of (rustling_result, wordseg_result).
    """
    results = []

    for name, cls in [("rustling", rustling_cls), ("wordseg", wordseg_cls)]:
        if cls is None:
            results.append(None)
            continue

        # Create and train model
        model = cls(max_word_length=max_word_length)
        model.fit(training_data)

        # Benchmark prediction
        # Note: wordseg returns a lazy map object, so we must consume it with list()
        def predict() -> None:
            result = model.predict(test_sentences)
            # Force evaluation of lazy iterators (wordseg returns map objects)
            list(result)

        total_time, _ = time_function(predict, iterations=iterations)

        avg_len = (
            sum(len(s) for s in test_sentences) // len(test_sentences)
            if test_sentences
            else 0
        )

        result = BenchmarkResult(
            name="LongestStringMatching",
            implementation=name,
            num_sentences=len(test_sentences),
            avg_sentence_length=avg_len,
            time_seconds=total_time,
            iterations=iterations,
        )
        results.append(result)

    return results[0], results[1]


def benchmark_random_segmenter(
    rustling_cls: type | None,
    wordseg_cls: type | None,
    test_sentences: list[str],
    prob: float = 0.5,
    iterations: int = 5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark RandomSegmenter implementations.

    Args:
        rustling_cls: rustling.wordseg.RandomSegmenter class.
        wordseg_cls: wordseg.RandomSegmenter class.
        test_sentences: Test sentences to segment.
        prob: Segmentation probability.
        iterations: Number of iterations for timing.

    Returns:
        Tuple of (rustling_result, wordseg_result).
    """
    results = []

    for name, cls in [("rustling", rustling_cls), ("wordseg", wordseg_cls)]:
        if cls is None:
            results.append(None)
            continue

        # Create model
        model = cls(prob=prob)

        # Benchmark prediction
        # Note: wordseg returns a lazy map object, so we must consume it with list()
        def predict() -> None:
            result = model.predict(test_sentences)
            # Force evaluation of lazy iterators (wordseg returns map objects)
            list(result)

        total_time, _ = time_function(predict, iterations=iterations)

        avg_len = (
            sum(len(s) for s in test_sentences) // len(test_sentences)
            if test_sentences
            else 0
        )

        result = BenchmarkResult(
            name="RandomSegmenter",
            implementation=name,
            num_sentences=len(test_sentences),
            avg_sentence_length=avg_len,
            time_seconds=total_time,
            iterations=iterations,
        )
        results.append(result)

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
    comparison = ComparisonResult(
        benchmark_name=rustling_result.name,
        rustling_time=rustling_result.time_seconds,
        wordseg_time=wordseg_result.time_seconds,
    )
    print(f"\n  ⚡ Speedup: {comparison.speedup:.1f}x faster")
    if comparison.speedup > 1:
        speedup = comparison.speedup
        print(f"     rustling is {speedup:.1f}x faster than pure Python wordseg")
    elif comparison.speedup < 1:
        print(f"     wordseg is {1/comparison.speedup:.1f}x faster than rustling")
    else:
        print("     Both implementations have similar performance")


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks.

    Args:
        quick: If True, run quick benchmarks with smaller data.
        verbose: If True, print results.

    Returns:
        Dictionary of benchmark results.
    """
    # Try to import both implementations
    rustling_lsm = None
    rustling_rs = None
    wordseg_lsm = None
    wordseg_rs = None

    try:
        from rustling.wordseg import LongestStringMatching as RustlingLSM
        from rustling.wordseg import RandomSegmenter as RustlingRS

        rustling_lsm = RustlingLSM
        rustling_rs = RustlingRS
        if verbose:
            print("✓ rustling.wordseg loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"✗ rustling.wordseg not available: {e}")

    try:
        from wordseg import LongestStringMatching as WordsegLSM
        from wordseg import RandomSegmenter as WordsegRS

        wordseg_lsm = WordsegLSM
        wordseg_rs = WordsegRS
        if verbose:
            print("✓ wordseg (pure Python) loaded successfully")
    except ImportError as e:
        if verbose:
            print(f"✗ wordseg (pure Python) not available: {e}")
            print("  Install with: pip install wordseg")

    if rustling_lsm is None and wordseg_lsm is None:
        print("\nError: Neither implementation is available. Cannot run benchmarks.")
        sys.exit(1)

    # Benchmark configurations
    if quick:
        configs = [
            {"num_words": 100, "num_sentences": 100, "iterations": 3},
        ]
    else:
        configs = [
            {"num_words": 100, "num_sentences": 100, "iterations": 5},
            {"num_words": 500, "num_sentences": 500, "iterations": 5},
            {"num_words": 1000, "num_sentences": 1000, "iterations": 5},
            {"num_words": 2000, "num_sentences": 2000, "iterations": 5},
            {"num_words": 5000, "num_sentences": 5000, "iterations": 3},
        ]

    all_results = {"benchmarks": [], "summary": {}}

    print("\n" + "=" * 60)
    print("WORDSEG BENCHMARK: rustling (Rust) vs wordseg (Python)")
    print("=" * 60)

    for config in configs:
        num_words = config["num_words"]
        num_sentences = config["num_sentences"]
        iterations = config["iterations"]

        if verbose:
            print(f"\n--- Config: {num_words} vocab, {num_sentences} sentences ---")

        # Generate data
        training_data = generate_training_data(num_words=num_words)
        test_sentences = generate_test_sentences(
            training_data, num_sentences=num_sentences
        )

        # Benchmark LongestStringMatching
        if verbose:
            print("\n📊 LongestStringMatching:")

        lsm_rustling, lsm_wordseg = benchmark_longest_string_matching(
            rustling_lsm,
            wordseg_lsm,
            training_data,
            test_sentences,
            iterations=iterations,
        )

        if lsm_rustling and verbose:
            print_result(lsm_rustling)
        if lsm_wordseg and verbose:
            print_result(lsm_wordseg)
        if lsm_rustling and lsm_wordseg and verbose:
            print_comparison(lsm_rustling, lsm_wordseg)

        # Benchmark RandomSegmenter
        if verbose:
            print("\n📊 RandomSegmenter:")

        rs_rustling, rs_wordseg = benchmark_random_segmenter(
            rustling_rs,
            wordseg_rs,
            test_sentences,
            iterations=iterations,
        )

        if rs_rustling and verbose:
            print_result(rs_rustling)
        if rs_wordseg and verbose:
            print_result(rs_wordseg)
        if rs_rustling and rs_wordseg and verbose:
            print_comparison(rs_rustling, rs_wordseg)

        # Collect results
        config_results = {
            "config": config,
            "LongestStringMatching": {
                "rustling": lsm_rustling.__dict__ if lsm_rustling else None,
                "wordseg": lsm_wordseg.__dict__ if lsm_wordseg else None,
            },
            "RandomSegmenter": {
                "rustling": rs_rustling.__dict__ if rs_rustling else None,
                "wordseg": rs_wordseg.__dict__ if rs_wordseg else None,
            },
        }
        all_results["benchmarks"].append(config_results)

    # Calculate summary
    lsm_speedups = []
    rs_speedups = []

    for bench in all_results["benchmarks"]:
        lsm = bench["LongestStringMatching"]
        if lsm["rustling"] and lsm["wordseg"]:
            speedup = lsm["wordseg"]["time_seconds"] / lsm["rustling"]["time_seconds"]
            lsm_speedups.append(speedup)

        rs = bench["RandomSegmenter"]
        if rs["rustling"] and rs["wordseg"]:
            speedup = rs["wordseg"]["time_seconds"] / rs["rustling"]["time_seconds"]
            rs_speedups.append(speedup)

    if lsm_speedups:
        all_results["summary"]["LongestStringMatching"] = {
            "avg_speedup": statistics.mean(lsm_speedups),
            "min_speedup": min(lsm_speedups),
            "max_speedup": max(lsm_speedups),
        }

    if rs_speedups:
        all_results["summary"]["RandomSegmenter"] = {
            "avg_speedup": statistics.mean(rs_speedups),
            "min_speedup": min(rs_speedups),
            "max_speedup": max(rs_speedups),
        }

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if "LongestStringMatching" in all_results["summary"]:
            s = all_results["summary"]["LongestStringMatching"]
            print("\nLongestStringMatching:")
            print(f"  Average speedup: {s['avg_speedup']:.1f}x")
            print(f"  Range: {s['min_speedup']:.1f}x - {s['max_speedup']:.1f}x")

        if "RandomSegmenter" in all_results["summary"]:
            s = all_results["summary"]["RandomSegmenter"]
            print("\nRandomSegmenter:")
            print(f"  Average speedup: {s['avg_speedup']:.1f}x")
            print(f"  Range: {s['min_speedup']:.1f}x - {s['max_speedup']:.1f}x")

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
        "--full",
        action="store_true",
        help="Run full benchmark suite (default)",
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

    # Set random seed for reproducibility
    random.seed(42)

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
