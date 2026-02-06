#!/usr/bin/env python
"""Benchmark with realistic Cantonese corpus data.

This script benchmarks POS tagging using realistic text data from
the HKCanCor corpus via pycantonese.

Usage:
    python benchmarks/taggers/realistic.py
    python benchmarks/taggers/realistic.py --quick
    python benchmarks/taggers/realistic.py --full
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any


def load_hkcancor_data() -> tuple[list[list[tuple[str, str]]], list[list[str]]]:
    """Load tagged sentences from HKCanCor.

    Returns:
        Tuple of (training_data, test_sentences)
    """
    try:
        import pycantonese
    except ImportError:
        raise ImportError(
            "pycantonese is required for realistic benchmarks. "
            "Install with: pip install pycantonese"
        )

    print("Loading HKCanCor corpus...")
    corpus = pycantonese.hkcancor()

    # Get tagged sentences
    # tokens() returns Token objects with word and pos attributes
    tagged_sents = []
    for sent_tokens in corpus.tokens(by_utterances=True):
        sent = [(token.word, token.pos) for token in sent_tokens if token.pos]
        if len(sent) >= 3:  # Filter out very short sentences
            tagged_sents.append(sent)

    print(f"Loaded {len(tagged_sents)} sentences from HKCanCor")

    # Split into train/test (80/20)
    split_idx = int(len(tagged_sents) * 0.8)
    training_data = tagged_sents[:split_idx]
    test_data = tagged_sents[split_idx:]

    # Test sentences are just words (no tags)
    test_sentences = [[word for word, tag in sent] for sent in test_data]

    print(f"Training sentences: {len(training_data)}")
    print(f"Test sentences: {len(test_sentences)}")

    return training_data, test_sentences


def benchmark_implementation(
    name: str,
    model_class: type,
    training_data: list[list[tuple[str, str]]],
    test_sentences: list[list[str]],
    train_iterations: int = 1,
    tag_iterations: int = 3,
) -> dict[str, Any]:
    """Benchmark a single implementation.

    Returns:
        Dictionary with timing results.
    """
    results = {"name": name}

    # Training
    print(f"\n  {name}:")
    print(f"    Training ({train_iterations} iteration(s))...", end=" ", flush=True)

    train_times = []
    for _ in range(train_iterations):
        gc.collect()
        model = model_class(frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5)
        start = time.perf_counter()
        model.train(training_data)
        train_times.append(time.perf_counter() - start)

    avg_train_time = sum(train_times) / len(train_times)
    print(f"{avg_train_time:.2f}s")
    results["train_time"] = avg_train_time

    # Train final model for tagging
    model = model_class(frequency_threshold=10, ambiguity_threshold=0.95, n_iter=5)
    model.train(training_data)

    # Tagging
    print(f"    Tagging ({tag_iterations} iteration(s))...", end=" ", flush=True)

    tag_times = []
    for _ in range(tag_iterations):
        gc.collect()
        start = time.perf_counter()
        for sent in test_sentences:
            model.tag(sent)
        tag_times.append(time.perf_counter() - start)

    avg_tag_time = sum(tag_times) / len(tag_times)
    sents_per_sec = len(test_sentences) / avg_tag_time
    print(f"{avg_tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")
    results["tag_time"] = avg_tag_time
    results["sents_per_sec"] = sents_per_sec

    return results


def run_benchmark(quick: bool = False) -> None:
    """Run the realistic benchmark."""
    print("=" * 60)
    print("Realistic POS Tagging Benchmark (HKCanCor)")
    print("=" * 60)

    # Load data
    try:
        training_data, test_sentences = load_hkcancor_data()
    except ImportError as e:
        print(f"Error: {e}")
        return

    if quick:
        # Use smaller subset for quick test
        training_data = training_data[:1000]
        test_sentences = test_sentences[:200]
        train_iterations = 1
        tag_iterations = 2
        n_train = len(training_data)
        n_test = len(test_sentences)
        print(f"\nQuick mode: using {n_train} train, {n_test} test")
    else:
        train_iterations = 1
        tag_iterations = 3

    results = []

    # Benchmark rustling
    try:
        from rustling.taggers import AveragedPerceptronTagger as RustlingTagger

        result = benchmark_implementation(
            "rustling.AveragedPerceptronTagger",
            RustlingTagger,
            training_data,
            test_sentences,
            train_iterations,
            tag_iterations,
        )
        results.append(result)
    except ImportError:
        print("\n  rustling not available, skipping...")

    # Benchmark pycantonese
    try:
        from pycantonese.pos_tagging.tagger import POSTagger as PycantoneseTagger

        result = benchmark_implementation(
            "pycantonese.POSTagger",
            PycantoneseTagger,
            training_data,
            test_sentences,
            train_iterations,
            tag_iterations,
        )
        results.append(result)
    except ImportError:
        print("\n  pycantonese POSTagger not available, skipping...")

    # Print comparison
    if len(results) == 2:
        rustling = results[0]
        pycantonese = results[1]

        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)

        train_speedup = pycantonese["train_time"] / rustling["train_time"]
        tag_speedup = pycantonese["tag_time"] / rustling["tag_time"]

        print(f"Training speedup: {train_speedup:.2f}x faster")
        print(f"Tagging speedup:  {tag_speedup:.2f}x faster")

        print("\nDetailed Results:")
        print(f"  {'Metric':<25} {'rustling':<15} {'pycantonese':<15}")
        print(f"  {'-' * 55}")
        r_train = rustling["train_time"]
        p_train = pycantonese["train_time"]
        r_tag = rustling["tag_time"]
        p_tag = pycantonese["tag_time"]
        r_sps = rustling["sents_per_sec"]
        p_sps = pycantonese["sents_per_sec"]
        print(f"  {'Training time (s)':<25} {r_train:<15.2f} {p_train:<15.2f}")
        print(f"  {'Tagging time (s)':<25} {r_tag:<15.4f} {p_tag:<15.4f}")
        print(f"  {'Sentences/second':<25} {r_sps:<15.0f} {p_sps:<15.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark POS taggers with realistic HKCanCor data"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark with smaller data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full benchmark",
    )
    args = parser.parse_args()

    # Default to quick if neither specified
    quick = args.quick or not args.full

    run_benchmark(quick=quick)


if __name__ == "__main__":
    main()
