#!/usr/bin/env python
"""Benchmark rustling.taggers vs pycantonese POS tagger.

This script compares the performance of rustling's Rust-based AveragedPerceptronTagger
against pycantonese's pure Python POSTagger implementation.

Usage:
    python benchmarks/taggers/run_tagger.py
    python benchmarks/taggers/run_tagger.py --quick      # Quick sanity check
    python benchmarks/taggers/run_tagger.py --full       # Full benchmark suite
    python benchmarks/taggers/run_tagger.py --export     # Export results to JSON
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    implementation: str
    num_sentences: int
    avg_sentence_length: float
    time_seconds: float
    iterations: int
    sentences_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        total_sentences = self.num_sentences * self.iterations
        self.sentences_per_second = total_sentences / self.time_seconds


@dataclass
class ComparisonResult:
    """Comparison between rustling and pycantonese."""

    benchmark_name: str
    rustling_time: float
    pycantonese_time: float
    speedup: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rustling_time > 0:
            self.speedup = self.pycantonese_time / self.rustling_time
        else:
            self.speedup = float("inf")


# Sample Cantonese words with their POS tags (based on HKCanCor tagset mapped to UD)
SAMPLE_TAGGED_DATA = [
    # Common sentence patterns
    [("我", "PRON"), ("係", "VERB"), ("學生", "NOUN"), ("。", "PUNCT")],
    [
        ("佢", "PRON"),
        ("好", "ADV"),
        ("鍾意", "VERB"),
        ("食", "VERB"),
        ("嘢", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("你", "PRON"),
        ("今日", "NOUN"),
        ("有冇", "VERB"),
        ("時間", "NOUN"),
        ("？", "PUNCT"),
    ],
    [
        ("我哋", "PRON"),
        ("一齊", "ADV"),
        ("去", "VERB"),
        ("睇", "VERB"),
        ("戲", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("呢", "DET"),
        ("本", "NOUN"),
        ("書", "NOUN"),
        ("好", "ADV"),
        ("好睇", "ADJ"),
        ("。", "PUNCT"),
    ],
    [
        ("佢哋", "PRON"),
        ("琴日", "NOUN"),
        ("返咗", "VERB"),
        ("屋企", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("我", "PRON"),
        ("想", "VERB"),
        ("飲", "VERB"),
        ("杯", "NOUN"),
        ("茶", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("呢度", "PRON"),
        ("嘅", "PART"),
        ("風景", "NOUN"),
        ("好", "ADV"),
        ("靚", "ADJ"),
        ("。", "PUNCT"),
    ],
    [
        ("你", "PRON"),
        ("識唔識", "VERB"),
        ("講", "VERB"),
        ("廣東話", "NOUN"),
        ("？", "PUNCT"),
    ],
    [
        ("我", "PRON"),
        ("啱啱", "ADV"),
        ("食", "VERB"),
        ("完", "VERB"),
        ("飯", "NOUN"),
        ("。", "PUNCT"),
    ],
    # More complex patterns
    [
        ("香港", "PROPN"),
        ("係", "VERB"),
        ("一", "NUM"),
        ("個", "NOUN"),
        ("好", "ADV"),
        ("繁華", "ADJ"),
        ("嘅", "PART"),
        ("城市", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("佢", "PRON"),
        ("每日", "NOUN"),
        ("都", "ADV"),
        ("要", "VERB"),
        ("返工", "VERB"),
        ("。", "PUNCT"),
    ],
    [
        ("我", "PRON"),
        ("聽日", "NOUN"),
        ("會", "AUX"),
        ("去", "VERB"),
        ("買", "VERB"),
        ("餸", "NOUN"),
        ("。", "PUNCT"),
    ],
    [
        ("呢", "DET"),
        ("間", "NOUN"),
        ("餐廳", "NOUN"),
        ("嘅", "PART"),
        ("嘢", "NOUN"),
        ("食", "VERB"),
        ("好", "ADV"),
        ("正", "ADJ"),
        ("。", "PUNCT"),
    ],
    [
        ("你", "PRON"),
        ("而家", "ADV"),
        ("喺", "VERB"),
        ("邊度", "PRON"),
        ("？", "PUNCT"),
    ],
]

# Vocabulary for generating more training data
VOCAB_BY_TAG = {
    "PRON": ["我", "你", "佢", "我哋", "你哋", "佢哋", "呢度", "嗰度", "邊度", "咩"],
    "VERB": [
        "係",
        "有",
        "冇",
        "去",
        "嚟",
        "食",
        "飲",
        "睇",
        "聽",
        "講",
        "做",
        "買",
        "賣",
        "返",
        "行",
        "坐",
        "企",
        "瞓",
    ],
    "NOUN": [
        "人",
        "嘢",
        "地方",
        "時間",
        "書",
        "筆",
        "車",
        "屋",
        "飯",
        "水",
        "茶",
        "錢",
        "日",
        "年",
        "學生",
        "老師",
    ],
    "ADJ": [
        "好",
        "靚",
        "大",
        "細",
        "長",
        "短",
        "新",
        "舊",
        "快",
        "慢",
        "平",
        "貴",
        "難",
        "易",
    ],
    "ADV": ["好", "都", "又", "先", "再", "仲", "已經", "啱啱", "而家", "一齊", "成日"],
    "DET": ["呢", "嗰", "邊", "每"],
    "PART": ["嘅", "咗", "緊", "過", "吖", "啦", "喎", "囉"],
    "PUNCT": ["。", "？", "！", "，"],
    "NUM": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    "PROPN": ["香港", "九龍", "新界", "中國", "日本", "美國"],
    "AUX": ["會", "可以", "應該", "要"],
}


def generate_training_data(
    num_sentences: int = 1000,
    min_words: int = 4,
    max_words: int = 12,
) -> list[list[tuple[str, str]]]:
    """Generate random tagged training sentences.

    Args:
        num_sentences: Total number of sentences to generate.
        min_words: Minimum words per sentence.
        max_words: Maximum words per sentence.

    Returns:
        A list of tagged sentences.
    """
    # Start with the sample data
    sentences = list(SAMPLE_TAGGED_DATA)

    # Common sentence patterns (tag sequences)
    patterns = [
        ["PRON", "VERB", "NOUN", "PUNCT"],
        ["PRON", "ADV", "VERB", "NOUN", "PUNCT"],
        ["PRON", "VERB", "VERB", "NOUN", "PUNCT"],
        ["DET", "NOUN", "NOUN", "ADV", "ADJ", "PUNCT"],
        ["PRON", "NOUN", "VERB", "NOUN", "PUNCT"],
        ["PROPN", "VERB", "NUM", "NOUN", "ADJ", "PART", "NOUN", "PUNCT"],
    ]

    while len(sentences) < num_sentences:
        # Select a pattern and generate a sentence
        pattern = random.choice(patterns)
        sentence = []
        for tag in pattern:
            if tag in VOCAB_BY_TAG:
                word = random.choice(VOCAB_BY_TAG[tag])
                sentence.append((word, tag))
        sentences.append(sentence)

    return sentences[:num_sentences]


def generate_test_sentences(
    training_data: list[list[tuple[str, str]]],
    num_sentences: int = 100,
) -> list[list[str]]:
    """Generate test sentences (just words, no tags).

    Args:
        training_data: Training data to sample vocabulary from.
        num_sentences: Number of test sentences to generate.

    Returns:
        A list of word sequences (untagged sentences).
    """
    # Extract all words from training data
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

    Args:
        model_class: The tagger class to benchmark.
        training_data: Training data.
        iterations: Number of iterations.

    Returns:
        Average training time in seconds.
    """
    times = []
    for _ in range(iterations):
        gc.collect()
        model = model_class(frequency_threshold=5, ambiguity_threshold=0.9, n_iter=3)
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

    Args:
        model: The trained tagger.
        test_sentences: Test sentences to tag.
        iterations: Number of iterations.

    Returns:
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


def run_benchmark(
    name: str,
    num_train_sentences: int,
    num_test_sentences: int,
    iterations: int = 3,
) -> dict[str, Any]:
    """Run a complete benchmark comparing both implementations.

    Args:
        name: Benchmark name.
        num_train_sentences: Number of training sentences.
        num_test_sentences: Number of test sentences.
        iterations: Number of iterations per measurement.

    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"  Training sentences: {num_train_sentences}")
    print(f"  Test sentences: {num_test_sentences}")
    print(f"{'=' * 60}")

    # Generate data
    training_data = generate_training_data(num_train_sentences)
    test_sentences = generate_test_sentences(training_data, num_test_sentences)

    avg_sent_len = sum(len(s) for s in training_data) / len(training_data)
    print(f"  Average sentence length: {avg_sent_len:.1f} words")

    results = {
        "name": name,
        "num_train_sentences": num_train_sentences,
        "num_test_sentences": num_test_sentences,
        "avg_sentence_length": avg_sent_len,
    }

    # Try to import rustling
    try:
        from rustling.taggers import AveragedPerceptronTagger as RustlingTagger

        has_rustling = True
    except ImportError:
        print("  Warning: rustling not available")
        has_rustling = False

    # Try to import pycantonese POSTagger
    try:
        from pycantonese.pos_tagging.tagger import POSTagger as PycantoneseTagger

        has_pycantonese = True
    except ImportError as e:
        print(f"  Warning: pycantonese not available ({e})")
        has_pycantonese = False
    except Exception as e:
        print(f"  Warning: pycantonese import error ({e})")
        has_pycantonese = False

    if not has_rustling and not has_pycantonese:
        print("  Error: Neither rustling nor pycantonese available!")
        return results

    # Benchmark rustling
    if has_rustling:
        print("\n  rustling.taggers.AveragedPerceptronTagger:")

        # Training
        train_time = benchmark_training(RustlingTagger, training_data, iterations)
        print(f"    Training time: {train_time:.4f}s")
        results["rustling_train_time"] = train_time

        # Create and train model for tagging benchmark
        model = RustlingTagger(frequency_threshold=5, ambiguity_threshold=0.9, n_iter=3)
        model.train(training_data)

        # Tagging
        tag_time = benchmark_tagging(model, test_sentences, iterations)
        sents_per_sec = num_test_sentences / tag_time
        print(f"    Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")
        results["rustling_tag_time"] = tag_time
        results["rustling_sents_per_sec"] = sents_per_sec

    # Benchmark pycantonese
    if has_pycantonese:
        print("\n  pycantonese.pos_tagging.POSTagger:")

        # Training
        train_time = benchmark_training(PycantoneseTagger, training_data, iterations)
        print(f"    Training time: {train_time:.4f}s")
        results["pycantonese_train_time"] = train_time

        # Create and train model for tagging benchmark
        model = PycantoneseTagger(
            frequency_threshold=5, ambiguity_threshold=0.9, n_iter=3
        )
        model.train(training_data)

        # Tagging
        tag_time = benchmark_tagging(model, test_sentences, iterations)
        sents_per_sec = num_test_sentences / tag_time
        print(f"    Tagging time: {tag_time:.4f}s ({sents_per_sec:.0f} sentences/sec)")
        results["pycantonese_tag_time"] = tag_time
        results["pycantonese_sents_per_sec"] = sents_per_sec

    # Calculate speedup
    if has_rustling and has_pycantonese:
        train_speedup = (
            results["pycantonese_train_time"] / results["rustling_train_time"]
        )
        tag_speedup = results["pycantonese_tag_time"] / results["rustling_tag_time"]
        print("\n  Speedup (rustling vs pycantonese):")
        print(f"    Training: {train_speedup:.2f}x faster")
        print(f"    Tagging:  {tag_speedup:.2f}x faster")
        results["train_speedup"] = train_speedup
        results["tag_speedup"] = tag_speedup

    return results


def run_quick_benchmark() -> list[dict[str, Any]]:
    """Run a quick sanity check benchmark."""
    print("\n" + "=" * 60)
    print("Quick Benchmark (sanity check)")
    print("=" * 60)

    results = []
    results.append(run_benchmark("Quick Test", 500, 100, iterations=2))
    return results


def run_full_benchmark() -> list[dict[str, Any]]:
    """Run the full benchmark suite."""
    print("\n" + "=" * 60)
    print("Full Benchmark Suite")
    print("=" * 60)

    results = []

    # Various data sizes
    configs = [
        ("Small", 500, 100),
        ("Medium", 2000, 500),
        ("Large", 5000, 1000),
        ("XLarge", 10000, 2000),
    ]

    for name, train_size, test_size in configs:
        results.append(run_benchmark(name, train_size, test_size, iterations=3))

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    # Header
    header1 = f"{'Benchmark':<15} {'Train':<12} {'Test':<12} {'Train':<12} {'Tag':<12}"
    header2 = f"{'Name':<15} {'Sentences':<12} {'Sentences':<12} {'Speedup':<12}"
    header2 += f" {'Speedup':<12}"
    print(header1)
    print(header2)
    print("-" * 80)

    for r in results:
        name = r.get("name", "Unknown")
        train_sents = r.get("num_train_sentences", 0)
        test_sents = r.get("num_test_sentences", 0)
        train_speedup = r.get("train_speedup", 0)
        tag_speedup = r.get("tag_speedup", 0)

        train_str = f"{train_speedup:.2f}x" if train_speedup else "N/A"
        tag_str = f"{tag_speedup:.2f}x" if tag_speedup else "N/A"

        row = f"{name:<15} {train_sents:<12} {test_sents:<12} {train_str:<12}"
        row += f" {tag_str:<12}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rustling.taggers vs pycantonese POS tagger"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick sanity check benchmark",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full benchmark suite",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export results to a JSON file",
    )
    args = parser.parse_args()

    # Default to quick if no option specified
    if not args.quick and not args.full:
        args.quick = True

    if args.quick:
        results = run_quick_benchmark()
    else:
        results = run_full_benchmark()

    print_summary(results)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to: {args.export}")


if __name__ == "__main__":
    main()
