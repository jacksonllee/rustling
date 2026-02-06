#!/usr/bin/env python
"""Benchmark POS taggers on English text: rustling, NLTK, and spaCy.

This script compares POS tagging performance on English text. It supports two modes:
- Training mode: Benchmark trainable models (rustling) on Brown corpus
- Inference mode: Benchmark all models including pre-trained (NLTK, spaCy, rustling)

Usage:
    python benchmarks/taggers/run_tagger_english.py --quick
    python benchmarks/taggers/run_tagger_english.py --mode inference
    python benchmarks/taggers/run_tagger_english.py --mode training
    python benchmarks/taggers/run_tagger_english.py --export results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.optional_imports import (  # noqa: E402
    print_library_status,
    try_import_nltk,
    try_import_spacy,
)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    implementation: str
    train_time: float | None
    tag_time: float
    num_train_sentences: int | None
    num_test_sentences: int
    sentences_per_second: float


def load_brown_corpus(
    train_limit: int = 40000,
    test_limit: int = 5000,
) -> tuple[list[list[tuple[str, str]]], list[list[str]]]:
    """Load Brown corpus data for training and testing.

    Parameters
    ----------
    train_limit : int, default=40000
        Number of training sentences to load
    test_limit : int, default=5000
        Number of test sentences to load

    Returns
    -------
    tuple[list[list[tuple[str, str]]], list[list[str]]]
        Tuple of (training_data, test_sentences).
        training_data: List of tagged sentences [(word, tag), ...]
        test_sentences: List of untagged sentences [word, ...]
    """
    try:
        from nltk.corpus import brown

        # Get tagged sentences with universal tagset
        tagged_sents = list(brown.tagged_sents(tagset="universal"))

        # Split into train and test
        train_data = tagged_sents[:train_limit]
        test_data = tagged_sents[train_limit : train_limit + test_limit]

        # Extract just words for test data
        test_sentences = [[word for word, tag in sent] for sent in test_data]

        return train_data, test_sentences

    except Exception as e:
        print(f"Error loading Brown corpus: {e}")
        print("Make sure NLTK is installed and Brown corpus is downloaded:")
        print("  python -m nltk.downloader brown universal_tagset")
        sys.exit(1)


def generate_english_training_data(
    num_sentences: int = 2000,
) -> list[list[tuple[str, str]]]:
    """Generate synthetic English training data.

    Parameters
    ----------
    num_sentences : int, default=2000
        Number of sentences to generate

    Returns
    -------
    list[list[tuple[str, str]]]
        List of tagged sentences
    """
    # English vocabulary by POS tag (Universal tagset)
    vocab_by_tag = {
        "NOUN": [
            "dog",
            "cat",
            "book",
            "car",
            "house",
            "tree",
            "apple",
            "computer",
            "friend",
            "teacher",
        ],
        "VERB": [
            "run",
            "walk",
            "eat",
            "sleep",
            "read",
            "write",
            "drive",
            "play",
            "work",
            "study",
        ],
        "ADJ": [
            "big",
            "small",
            "red",
            "blue",
            "happy",
            "sad",
            "fast",
            "slow",
            "good",
            "bad",
        ],
        "ADV": [
            "quickly",
            "slowly",
            "very",
            "really",
            "always",
            "never",
            "often",
            "sometimes",
        ],
        "PRON": ["I", "you", "he", "she", "it", "we", "they", "this", "that"],
        "DET": ["the", "a", "an", "this", "that", "these", "those", "my", "your"],
        "ADP": ["in", "on", "at", "to", "from", "with", "by", "for", "of"],
        "CONJ": ["and", "or", "but", "because", "if", "when", "while"],
        "PRT": ["up", "down", "out", "off", "over"],
        ".": [".", "!", "?"],
    }

    # Common sentence patterns
    patterns = [
        ["PRON", "VERB", "NOUN", "."],
        ["DET", "NOUN", "VERB", "ADV", "."],
        ["PRON", "VERB", "DET", "ADJ", "NOUN", "."],
        ["DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN", "."],
        ["PRON", "VERB", "CONJ", "PRON", "VERB", "NOUN", "."],
    ]

    sentences = []
    for _ in range(num_sentences):
        pattern = random.choice(patterns)
        sentence = []
        for tag in pattern:
            if tag in vocab_by_tag:
                word = random.choice(vocab_by_tag[tag])
                sentence.append((word, tag))
        sentences.append(sentence)

    return sentences


def generate_test_sentences(
    training_data: list[list[tuple[str, str]]],
    num_sentences: int = 500,
) -> list[list[str]]:
    """Generate test sentences from training vocabulary.

    Parameters
    ----------
    training_data : list[list[tuple[str, str]]]
        Training data to extract vocabulary from
    num_sentences : int, default=500
        Number of test sentences to generate

    Returns
    -------
    list[list[str]]
        List of word sequences (untagged)
    """
    # Extract all words
    all_words = []
    for sent in training_data:
        all_words.extend([word for word, tag in sent])

    unique_words = list(set(all_words))

    test_sentences = []
    for _ in range(num_sentences):
        sent_len = random.randint(4, 12)
        sent = random.choices(unique_words, k=sent_len)
        test_sentences.append(sent)

    return test_sentences


def benchmark_training(
    model_class: type,
    training_data: list[list[tuple[str, str]]],
    iterations: int = 3,
) -> float:
    """Benchmark training time.

    Parameters
    ----------
    model_class : type
        The tagger class to benchmark
    training_data : list[list[tuple[str, str]]]
        Training data
    iterations : int, default=3
        Number of iterations

    Returns
    -------
    float
        Average training time in seconds
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = model_class()
        start = time.perf_counter()
        model.train(training_data)
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


def benchmark_tagging(
    model: Any,
    test_sentences: list[list[str]],
    iterations: int = 5,
) -> float:
    """Benchmark tagging time.

    Parameters
    ----------
    model : Any
        The trained tagger
    test_sentences : list[list[str]]
        Test sentences to tag
    iterations : int, default=5
        Number of iterations

    Returns
    -------
    float
        Average tagging time in seconds
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        for sent in test_sentences:
            model.tag(sent)
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


def benchmark_nltk_tagging(
    pos_tag_sents: callable,
    test_sentences: list[list[str]],
    iterations: int = 5,
) -> float:
    """Benchmark NLTK tagging time.

    NLTK's pos_tag_sents expects a list of sentences (each sentence is a list of words).

    Parameters
    ----------
    pos_tag_sents : callable
        NLTK pos_tag_sents function
    test_sentences : list[list[str]]
        Test sentences to tag
    iterations : int, default=5
        Number of iterations

    Returns
    -------
    float
        Average tagging time in seconds
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        pos_tag_sents(test_sentences)
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


def benchmark_spacy_tagging(
    nlp: Any,
    test_sentences: list[list[str]],
    iterations: int = 5,
) -> float:
    """Benchmark spaCy tagging time.

    Parameters
    ----------
    nlp : Any
        spaCy Language model
    test_sentences : list[list[str]]
        Test sentences to tag
    iterations : int, default=5
        Number of iterations

    Returns
    -------
    float
        Average tagging time in seconds
    """
    # Join words into strings for spaCy
    test_texts = [" ".join(sent) for sent in test_sentences]

    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        for text in test_texts:
            doc = nlp(text)
            [(token.text, token.pos_) for token in doc]
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


def run_training_mode(
    num_train: int = 40000,
    num_test: int = 5000,
    use_brown: bool = True,
    iterations: int = 3,
) -> dict[str, Any]:
    """Run training mode benchmark (trainable models only).

    Parameters
    ----------
    num_train : int, default=40000
        Number of training sentences
    num_test : int, default=5000
        Number of test sentences
    use_brown : bool, default=True
        If True, use Brown corpus; otherwise generate synthetic data
    iterations : int, default=3
        Number of iterations per measurement

    Returns
    -------
    dict[str, Any]
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("TRAINING MODE: Trainable Models")
    print("=" * 60)

    # Load data
    if use_brown:
        print(f"\nLoading Brown corpus (train: {num_train}, test: {num_test})...")
        training_data, test_sentences = load_brown_corpus(num_train, num_test)
        print(f"Loaded {len(training_data)} training sentences")
        print(f"Loaded {len(test_sentences)} test sentences")
    else:
        print(f"\nGenerating synthetic data (train: {num_train}, test: {num_test})...")
        training_data = generate_english_training_data(num_train)
        test_sentences = generate_test_sentences(training_data, num_test)

    results = {
        "mode": "training",
        "num_train_sentences": len(training_data),
        "num_test_sentences": len(test_sentences),
        "benchmarks": {},
    }

    # Try to import rustling
    try:
        from rustling.taggers import AveragedPerceptronTagger

        has_rustling = True
    except ImportError:
        print("\n⚠️  rustling not available")
        has_rustling = False

    if not has_rustling:
        print("\nNo trainable models available for training mode benchmark.")
        return results

    # Benchmark rustling
    print("\n📊 rustling.taggers.AveragedPerceptronTagger:")

    # Training
    print(f"  Training on {len(training_data)} sentences...")
    train_time = benchmark_training(AveragedPerceptronTagger, training_data, iterations)
    print(f"  Training time: {train_time:.4f}s")

    # Create and train model for tagging
    model = AveragedPerceptronTagger()
    model.train(training_data)

    # Tagging
    print(f"  Tagging {len(test_sentences)} sentences...")
    tag_time = benchmark_tagging(model, test_sentences, iterations)
    sents_per_sec = len(test_sentences) / tag_time
    print(f"  Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")

    results["benchmarks"]["rustling"] = {
        "implementation": "rustling.AveragedPerceptronTagger",
        "train_time": train_time,
        "tag_time": tag_time,
        "sentences_per_second": sents_per_sec,
    }

    return results


def run_inference_mode(
    num_test: int = 2000,
    use_brown: bool = True,
    iterations: int = 5,
) -> dict[str, Any]:
    """Run inference mode benchmark (all models, including pre-trained).

    Parameters
    ----------
    num_test : int, default=2000
        Number of test sentences
    use_brown : bool, default=True
        If True, use Brown corpus; otherwise generate synthetic data
    iterations : int, default=5
        Number of iterations per measurement

    Returns
    -------
    dict[str, Any]
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("INFERENCE MODE: All Models (Pre-trained + Trainable)")
    print("=" * 60)

    # Import libraries
    nltk_funcs = try_import_nltk()
    spacy_funcs = try_import_spacy()

    print()
    print_library_status("NLTK", nltk_funcs, verbose=True)
    print_library_status("spaCy", spacy_funcs, verbose=True)

    # Load data
    if use_brown and nltk_funcs["available"]:
        print(f"\nLoading Brown corpus test data ({num_test} sentences)...")
        _, test_sentences = load_brown_corpus(train_limit=40000, test_limit=num_test)
    else:
        print(f"\nGenerating synthetic test data ({num_test} sentences)...")
        training_data = generate_english_training_data(1000)
        test_sentences = generate_test_sentences(training_data, num_test)

    print(f"Test sentences: {len(test_sentences)}")

    results = {
        "mode": "inference",
        "num_test_sentences": len(test_sentences),
        "benchmarks": {},
    }

    # Try to import rustling
    try:
        from rustling.taggers import AveragedPerceptronTagger

        has_rustling = True
    except ImportError:
        print("\n⚠️  rustling not available")
        has_rustling = False

    # Benchmark rustling
    if has_rustling:
        print("\n📊 rustling.taggers.AveragedPerceptronTagger:")
        print("  Note: Using model trained on generated data (not Brown corpus)")

        # Train a quick model
        train_data = generate_english_training_data(5000)
        model = AveragedPerceptronTagger()
        model.train(train_data)

        # Benchmark tagging
        tag_time = benchmark_tagging(model, test_sentences, iterations)
        sents_per_sec = len(test_sentences) / tag_time
        print(f"  Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")

        results["benchmarks"]["rustling"] = {
            "implementation": "rustling.AveragedPerceptronTagger",
            "tag_time": tag_time,
            "sentences_per_second": sents_per_sec,
            "note": "Trained on synthetic data",
        }

    # Benchmark NLTK
    if nltk_funcs["available"]:
        print("\n📊 NLTK pos_tag:")
        print("  Note: Pre-trained on Penn Treebank")

        pos_tag_sents = nltk_funcs["pos_tag_sents"]
        tag_time = benchmark_nltk_tagging(pos_tag_sents, test_sentences, iterations)
        sents_per_sec = len(test_sentences) / tag_time
        print(f"  Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")

        results["benchmarks"]["nltk"] = {
            "implementation": "NLTK pos_tag",
            "tag_time": tag_time,
            "sentences_per_second": sents_per_sec,
            "note": "Pre-trained on Penn Treebank",
        }

    # Benchmark spaCy
    if spacy_funcs["available"]:
        print(f"\n📊 spaCy ({spacy_funcs['model_name']}):")
        print("  Note: Pre-trained transformer model")

        nlp = spacy_funcs["nlp"]
        tag_time = benchmark_spacy_tagging(nlp, test_sentences, iterations)
        sents_per_sec = len(test_sentences) / tag_time
        print(f"  Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")

        results["benchmarks"]["spacy"] = {
            "implementation": f"spaCy {spacy_funcs['model_name']}",
            "tag_time": tag_time,
            "sentences_per_second": sents_per_sec,
            "note": "Pre-trained transformer model",
        }

    # Print comparison
    if len(results["benchmarks"]) >= 2:
        print("\n" + "=" * 60)
        print("SPEEDUP COMPARISON")
        print("=" * 60)

        # Find fastest
        fastest_name = min(
            results["benchmarks"].keys(),
            key=lambda k: results["benchmarks"][k]["tag_time"],
        )
        fastest_impl = results["benchmarks"][fastest_name]["implementation"]
        fastest_time = results["benchmarks"][fastest_name]["tag_time"]

        print(f"\nFastest: {fastest_impl} ({fastest_time:.4f}s)")
        print("\nRelative performance:")
        for name, bench in results["benchmarks"].items():
            if name != fastest_name:
                speedup = bench["tag_time"] / fastest_time
                print(f"  {bench['implementation']}: {speedup:.2f}x slower")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark POS taggers on English text"
    )
    parser.add_argument(
        "--mode",
        choices=["training", "inference", "both"],
        default="inference",
        help="Benchmark mode: training (trainable only) or inference (all models)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller data",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of Brown corpus",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Configure sizes
    if args.quick:
        num_train = 2000
        num_test = 500
        iterations_train = 2
        iterations_tag = 3
    else:
        num_train = 40000
        num_test = 2000
        iterations_train = 3
        iterations_tag = 5

    use_brown = not args.synthetic

    all_results = {"mode": args.mode, "results": {}}

    # Run benchmarks
    if args.mode in ("training", "both"):
        train_results = run_training_mode(
            num_train=num_train,
            num_test=num_test,
            use_brown=use_brown,
            iterations=iterations_train,
        )
        all_results["results"]["training"] = train_results

    if args.mode in ("inference", "both"):
        inference_results = run_inference_mode(
            num_test=num_test,
            use_brown=use_brown,
            iterations=iterations_tag,
        )
        all_results["results"]["inference"] = inference_results

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        with open(export_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
