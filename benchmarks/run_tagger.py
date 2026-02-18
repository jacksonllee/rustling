#!/usr/bin/env python
"""Benchmark rustling.tagging vs NLTK PerceptronTagger.

Compares training and tagging speed using HKCanCor corpus data.

Usage:
    python benchmarks/run_tagger.py
    python benchmarks/run_tagger.py --quick
    python benchmarks/run_tagger.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common.data import load_hkcancor, tagging_data  # noqa: E402


def try_import_rustling_tagger() -> dict[str, Any]:
    """Try to import Rustling's AveragedPerceptronTagger."""
    try:
        from rustling.tagging import AveragedPerceptronTagger

        return {"available": True, "class": AveragedPerceptronTagger}
    except ImportError as e:
        return {"available": False, "error": str(e)}


def try_import_nltk_tagger() -> dict[str, Any]:
    """Try to import NLTK's PerceptronTagger."""
    try:
        from nltk.tag import PerceptronTagger

        return {"available": True, "class": PerceptronTagger}
    except ImportError as e:
        return {"available": False, "error": str(e)}


def benchmark_rustling_training(
    tagger_class: type,
    training_data: list[list[tuple[str, str]]],
    iterations: int = 3,
) -> float:
    """Benchmark Rustling training time.

    Returns
    -------
    float
        Average training time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = tagger_class()
        start = time.perf_counter()
        model.fit(training_data)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_rustling_tagging(
    model: Any,
    test_sentences: list[list[str]],
    iterations: int = 5,
) -> float:
    """Benchmark Rustling tagging time.

    Returns
    -------
    float
        Average tagging time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        for sent in test_sentences:
            model.predict(sent)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_nltk_training(
    tagger_class: type,
    training_data: list[list[tuple[str, str]]],
    iterations: int = 3,
) -> float:
    """Benchmark NLTK PerceptronTagger training time.

    Returns
    -------
    float
        Average training time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = tagger_class(load=False)
        start = time.perf_counter()
        model.train(training_data)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def benchmark_nltk_tagging(
    model: Any,
    test_sentences: list[list[str]],
    iterations: int = 5,
) -> float:
    """Benchmark NLTK PerceptronTagger tagging time.

    Returns
    -------
    float
        Average tagging time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        for sent in test_sentences:
            model.tag(sent)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


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
    rustling_info = try_import_rustling_tagger()
    nltk_info = try_import_nltk_tagger()

    if verbose:
        if rustling_info["available"]:
            print("✓ rustling.tagging loaded successfully")
        else:
            print(f"✗ rustling.tagging not available: {rustling_info.get('error', '')}")
        if nltk_info["available"]:
            print("✓ NLTK PerceptronTagger loaded successfully")
        else:
            print(f"✗ NLTK not available: {nltk_info.get('error', '')}")

    if not rustling_info["available"] and not nltk_info["available"]:
        print("\nError: Neither implementation is available.")
        sys.exit(1)

    # Load data
    if verbose:
        print("\nLoading HKCanCor corpus...")
    tagged_sents = load_hkcancor()
    training_data, test_sentences = tagging_data(tagged_sents)

    if quick:
        training_data = training_data[:1000]
        test_sentences = test_sentences[:200]
        train_iterations = 2
        tag_iterations = 3
    else:
        train_iterations = 3
        tag_iterations = 5

    if verbose:
        print(f"Training sentences: {len(training_data)}")
        print(f"Test sentences: {len(test_sentences)}")

    results: dict[str, Any] = {
        "num_train": len(training_data),
        "num_test": len(test_sentences),
        "benchmarks": {},
    }

    print(
        "\n" + "=" * 70 + "\nPOS TAGGER BENCHMARK:"
        " Rustling (Rust) vs NLTK PerceptronTagger (Python)" + "\n" + "=" * 70
    )

    # --- Training ---
    print(f"\n--- Training ({train_iterations} iterations) ---")

    rustling_train_time = None
    if rustling_info["available"]:
        rustling_train_time = benchmark_rustling_training(
            rustling_info["class"], training_data, train_iterations
        )
        if verbose:
            print(
                f"\n  rustling.tagging.AveragedPerceptronTagger:"
                f"\n    Training time: {rustling_train_time:.4f}s"
            )

    nltk_train_time = None
    if nltk_info["available"]:
        nltk_train_time = benchmark_nltk_training(
            nltk_info["class"], training_data, train_iterations
        )
        if verbose:
            print(
                f"\n  NLTK PerceptronTagger:"
                f"\n    Training time: {nltk_train_time:.4f}s"
            )

    if rustling_train_time and nltk_train_time:
        speedup = nltk_train_time / rustling_train_time
        print(f"\n  ⚡ Training speedup: {speedup:.1f}x faster")
        results["benchmarks"]["training"] = {
            "rustling": rustling_train_time,
            "nltk": nltk_train_time,
            "speedup": speedup,
        }

    # --- Tagging ---
    print(f"\n--- Tagging ({tag_iterations} iterations) ---")

    rustling_tag_time = None
    if rustling_info["available"]:
        model = rustling_info["class"]()
        model.fit(training_data)
        rustling_tag_time = benchmark_rustling_tagging(
            model, test_sentences, tag_iterations
        )
        sps = len(test_sentences) / rustling_tag_time
        if verbose:
            print(
                f"\n  rustling.tagging.AveragedPerceptronTagger:"
                f"\n    Tagging time: {rustling_tag_time:.4f}s"
                f" ({sps:,.0f} sentences/sec)"
            )
        results["benchmarks"].setdefault("tagging", {})["rustling"] = {
            "time": rustling_tag_time,
            "sentences_per_sec": sps,
        }

    nltk_tag_time = None
    if nltk_info["available"]:
        model = nltk_info["class"](load=False)
        model.train(training_data)
        nltk_tag_time = benchmark_nltk_tagging(model, test_sentences, tag_iterations)
        sps = len(test_sentences) / nltk_tag_time
        if verbose:
            print(
                f"\n  NLTK PerceptronTagger:"
                f"\n    Tagging time: {nltk_tag_time:.4f}s"
                f" ({sps:,.0f} sentences/sec)"
            )
        results["benchmarks"].setdefault("tagging", {})["nltk"] = {
            "time": nltk_tag_time,
            "sentences_per_sec": sps,
        }

    if rustling_tag_time and nltk_tag_time:
        speedup = nltk_tag_time / rustling_tag_time
        print(f"\n  ⚡ Tagging speedup: {speedup:.1f}x faster")
        results["benchmarks"]["tagging"]["speedup"] = speedup

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Rustling vs NLTK PerceptronTagger"
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
