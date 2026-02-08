#!/usr/bin/env python
"""Benchmark rustling.lm vs NLTK language models.

Compares fit, score, and generate operations using HKCanCor corpus data.

Usage:
    python benchmarks/run_lm.py
    python benchmarks/run_lm.py --quick
    python benchmarks/run_lm.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.data import lm_data, load_hkcancor  # noqa: E402


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    implementation: str
    operation: str
    num_items: int
    time_seconds: float
    iterations: int
    items_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        total_items = self.num_items * self.iterations
        self.items_per_second = (
            total_items / self.time_seconds if self.time_seconds > 0 else float("inf")
        )


@dataclass
class ComparisonResult:
    """Comparison between rustling and NLTK."""

    benchmark_name: str
    rustling_time: float
    nltk_time: float
    speedup: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rustling_time > 0:
            self.speedup = self.nltk_time / self.rustling_time
        else:
            self.speedup = float("inf")


def generate_score_pairs(
    training_data: list[list[str]],
    num_pairs: int = 1000,
    context_len: int = 1,
) -> list[tuple[str, list[str]]]:
    """Generate (word, context) pairs for scoring benchmarks."""
    all_words = [word for sent in training_data for word in sent]
    if not all_words:
        return []

    pairs = []
    for _ in range(num_pairs):
        word = random.choice(all_words)
        context = random.choices(all_words, k=context_len)
        pairs.append((word, context))
    return pairs


def time_function(
    func: Callable[[], Any],
    iterations: int = 5,
    warmup: int = 1,
) -> tuple[float, list[float]]:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()

    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return sum(times), times


def try_import_nltk_lm() -> dict[str, Any]:
    """Try to import NLTK language model components."""
    try:
        from nltk.lm import MLE, Laplace, Lidstone
        from nltk.lm.preprocessing import padded_everygram_pipeline

        return {
            "available": True,
            "MLE": MLE,
            "Lidstone": Lidstone,
            "Laplace": Laplace,
            "padded_everygram_pipeline": padded_everygram_pipeline,
        }
    except ImportError as e:
        return {
            "available": False,
            "error": f"NLTK not installed or nltk.lm unavailable: {e}",
        }


def try_import_rustling_lm() -> dict[str, Any]:
    """Try to import rustling language model components."""
    try:
        from rustling.lm import MLE, Laplace, Lidstone

        return {
            "available": True,
            "MLE": MLE,
            "Lidstone": Lidstone,
            "Laplace": Laplace,
        }
    except ImportError as e:
        return {
            "available": False,
            "error": f"rustling.lm not available: {e}",
        }


def benchmark_fit(
    rustling_info: dict[str, Any],
    nltk_info: dict[str, Any],
    training_data: list[list[str]],
    model_name: str,
    order: int,
    iterations: int,
    gamma: float = 0.5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark model training (fit)."""
    results = []

    # Rustling
    if rustling_info["available"]:
        cls = rustling_info[model_name]
        kwargs = {"order": order}
        if model_name == "Lidstone":
            kwargs["gamma"] = gamma

        def rustling_fit():
            model = cls(**kwargs)
            model.fit(training_data)

        total_time, _ = time_function(rustling_fit, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="rustling",
                operation="fit",
                num_items=len(training_data),
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    # NLTK
    if nltk_info["available"]:
        cls = nltk_info[model_name]
        pipeline = nltk_info["padded_everygram_pipeline"]

        nltk_kwargs = {}
        if model_name == "Lidstone":
            nltk_kwargs["gamma"] = gamma

        def nltk_fit():
            model = cls(order=order, **nltk_kwargs)
            train, vocab = pipeline(order, training_data)
            model.fit(train, vocab)

        total_time, _ = time_function(nltk_fit, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="nltk",
                operation="fit",
                num_items=len(training_data),
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    return results[0], results[1]


def benchmark_score(
    rustling_info: dict[str, Any],
    nltk_info: dict[str, Any],
    training_data: list[list[str]],
    score_pairs: list[tuple[str, list[str]]],
    model_name: str,
    order: int,
    iterations: int,
    gamma: float = 0.5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark scoring."""
    results = []

    # Rustling
    if rustling_info["available"]:
        cls = rustling_info[model_name]
        kwargs = {"order": order}
        if model_name == "Lidstone":
            kwargs["gamma"] = gamma
        model = cls(**kwargs)
        model.fit(training_data)

        def rustling_score():
            for word, context in score_pairs:
                model.score(word, context)

        total_time, _ = time_function(rustling_score, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="rustling",
                operation="score",
                num_items=len(score_pairs),
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    # NLTK
    if nltk_info["available"]:
        cls = nltk_info[model_name]
        pipeline = nltk_info["padded_everygram_pipeline"]

        nltk_kwargs = {}
        if model_name == "Lidstone":
            nltk_kwargs["gamma"] = gamma
        model = cls(order=order, **nltk_kwargs)
        train, vocab = pipeline(order, training_data)
        model.fit(train, vocab)

        def nltk_score():
            for word, context in score_pairs:
                model.score(word, tuple(context))

        total_time, _ = time_function(nltk_score, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="nltk",
                operation="score",
                num_items=len(score_pairs),
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    return results[0], results[1]


def benchmark_generate(
    rustling_info: dict[str, Any],
    nltk_info: dict[str, Any],
    training_data: list[list[str]],
    model_name: str,
    order: int,
    num_words: int,
    iterations: int,
    gamma: float = 0.5,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Benchmark text generation."""
    results = []

    # Rustling
    if rustling_info["available"]:
        cls = rustling_info[model_name]
        kwargs = {"order": order}
        if model_name == "Lidstone":
            kwargs["gamma"] = gamma
        model = cls(**kwargs)
        model.fit(training_data)

        seed_counter = [0]

        def rustling_generate():
            model.generate(num_words=num_words, random_seed=seed_counter[0])
            seed_counter[0] += 1

        total_time, _ = time_function(rustling_generate, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="rustling",
                operation="generate",
                num_items=num_words,
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    # NLTK
    if nltk_info["available"]:
        cls = nltk_info[model_name]
        pipeline = nltk_info["padded_everygram_pipeline"]

        nltk_kwargs = {}
        if model_name == "Lidstone":
            nltk_kwargs["gamma"] = gamma
        model = cls(order=order, **nltk_kwargs)
        train, vocab = pipeline(order, training_data)
        model.fit(train, vocab)

        seed_counter = [0]

        def nltk_generate():
            model.generate(num_words=num_words, random_seed=seed_counter[0])
            seed_counter[0] += 1

        total_time, _ = time_function(nltk_generate, iterations=iterations)
        results.append(
            BenchmarkResult(
                name=model_name,
                implementation="nltk",
                operation="generate",
                num_items=num_words,
                time_seconds=total_time,
                iterations=iterations,
            )
        )
    else:
        results.append(None)

    return results[0], results[1]


def print_result(result: BenchmarkResult) -> None:
    """Print a benchmark result."""
    print(f"  {result.implementation}:")
    print(
        f"    Total time: {result.time_seconds:.4f}s"
        f" ({result.iterations} iterations)"
    )
    print(f"    {result.operation}/second: {result.items_per_second:,.0f}")


def print_comparison(
    rustling_result: BenchmarkResult, nltk_result: BenchmarkResult
) -> None:
    """Print comparison between rustling and NLTK."""
    comparison = ComparisonResult(
        benchmark_name=f"{rustling_result.name} {rustling_result.operation}",
        rustling_time=rustling_result.time_seconds,
        nltk_time=nltk_result.time_seconds,
    )
    print(f"\n  ⚡ Speedup: {comparison.speedup:.1f}x faster")
    if comparison.speedup > 1:
        print(f"     rustling is {comparison.speedup:.1f}x faster than NLTK")
    elif comparison.speedup < 1:
        print(f"     NLTK is {1/comparison.speedup:.1f}x faster than rustling")
    else:
        print("     Both implementations have similar performance")


def run_benchmarks(
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all benchmarks."""
    rustling_info = try_import_rustling_lm()
    nltk_info = try_import_nltk_lm()

    if verbose:
        if rustling_info["available"]:
            print("✓ rustling.lm loaded successfully")
        else:
            print(f"✗ rustling.lm not available: {rustling_info.get('error', '')}")

        if nltk_info["available"]:
            print("✓ NLTK language models loaded successfully")
        else:
            print(f"✗ NLTK not available: {nltk_info.get('error', '')}")
            print("  Install with: pip install nltk")

    if not rustling_info["available"] and not nltk_info["available"]:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    # Load data
    if verbose:
        print("\nLoading HKCanCor corpus...")
    tagged_sents = load_hkcancor()
    training_data = lm_data(tagged_sents)

    order = 3

    if quick:
        training_data = training_data[:500]
        num_score_pairs = 1000
        num_generate_words = 100
        iterations = 3
    else:
        num_score_pairs = 5000
        num_generate_words = 500
        iterations = 5

    score_pairs = generate_score_pairs(
        training_data, num_pairs=num_score_pairs, context_len=order - 1
    )

    if verbose:
        print(f"Training sentences: {len(training_data)}")

    model_names = ["MLE", "Lidstone", "Laplace"]
    all_results: dict[str, Any] = {"benchmarks": [], "summary": {}}
    speedups_by_op: dict[str, list[float]] = {}

    print("\n" + "=" * 70)
    print("LANGUAGE MODEL BENCHMARK: Rustling (Rust) vs NLTK (Python)")
    print("=" * 70)

    if verbose:
        print(
            f"\nConfig: {len(training_data)} sentences, order={order},"
            f" {num_score_pairs} score pairs,"
            f" {num_generate_words} generate words"
        )

    config_results: dict[str, Any] = {"models": {}}

    for model_name in model_names:
        if verbose:
            print(f"\n📊 {model_name}:")

        model_results: dict[str, Any] = {}

        # --- fit ---
        if verbose:
            print(f"\n  [fit] Training on {len(training_data)} sentences:")

        r_fit, n_fit = benchmark_fit(
            rustling_info,
            nltk_info,
            training_data,
            model_name,
            order,
            iterations,
        )
        if r_fit and verbose:
            print_result(r_fit)
        if n_fit and verbose:
            print_result(n_fit)
        if r_fit and n_fit and verbose:
            print_comparison(r_fit, n_fit)
        if r_fit and n_fit:
            key = f"{model_name} fit"
            speedups_by_op.setdefault(key, []).append(
                n_fit.time_seconds / r_fit.time_seconds
                if r_fit.time_seconds > 0
                else float("inf")
            )
        model_results["fit"] = {
            "rustling": r_fit.__dict__ if r_fit else None,
            "nltk": n_fit.__dict__ if n_fit else None,
        }

        # --- score ---
        if verbose:
            print(f"\n  [score] Scoring {num_score_pairs} word/context pairs:")

        r_score, n_score = benchmark_score(
            rustling_info,
            nltk_info,
            training_data,
            score_pairs,
            model_name,
            order,
            iterations,
        )
        if r_score and verbose:
            print_result(r_score)
        if n_score and verbose:
            print_result(n_score)
        if r_score and n_score and verbose:
            print_comparison(r_score, n_score)
        if r_score and n_score:
            key = f"{model_name} score"
            speedups_by_op.setdefault(key, []).append(
                n_score.time_seconds / r_score.time_seconds
                if r_score.time_seconds > 0
                else float("inf")
            )
        model_results["score"] = {
            "rustling": r_score.__dict__ if r_score else None,
            "nltk": n_score.__dict__ if n_score else None,
        }

        # --- generate ---
        if verbose:
            print(f"\n  [generate] Generating {num_generate_words} words:")

        r_gen, n_gen = benchmark_generate(
            rustling_info,
            nltk_info,
            training_data,
            model_name,
            order,
            num_generate_words,
            iterations,
        )
        if r_gen and verbose:
            print_result(r_gen)
        if n_gen and verbose:
            print_result(n_gen)
        if r_gen and n_gen and verbose:
            print_comparison(r_gen, n_gen)
        if r_gen and n_gen:
            key = f"{model_name} generate"
            speedups_by_op.setdefault(key, []).append(
                n_gen.time_seconds / r_gen.time_seconds
                if r_gen.time_seconds > 0
                else float("inf")
            )
        model_results["generate"] = {
            "rustling": r_gen.__dict__ if r_gen else None,
            "nltk": n_gen.__dict__ if n_gen else None,
        }

        config_results["models"][model_name] = model_results

    all_results["benchmarks"].append(config_results)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for key, speedup_list in sorted(speedups_by_op.items()):
            avg = statistics.mean(speedup_list)
            print(f"\n  {key}: {avg:.1f}x faster")

    all_results["summary"] = {
        key: {"avg_speedup": statistics.mean(vals)}
        for key, vals in speedups_by_op.items()
    }

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.lm vs NLTK language models"
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
