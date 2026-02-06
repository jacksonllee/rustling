#!/usr/bin/env python
"""Scaling benchmark for POS taggers.

This script generates a performance scaling analysis, showing how
rustling and pycantonese taggers scale with increasing data sizes.

Usage:
    python benchmarks/taggers/scaling.py
    python benchmarks/taggers/scaling.py --plot  # Generate ASCII chart
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from dataclasses import dataclass


@dataclass
class ScalingPoint:
    """A single data point in the scaling benchmark."""

    size: int
    rustling_train_time: float | None
    rustling_tag_time: float | None
    pycantonese_train_time: float | None
    pycantonese_tag_time: float | None
    train_speedup: float | None
    tag_speedup: float | None


# Vocabulary for generating training data
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
    "PUNCT": ["。", "？", "！"],
}

PATTERNS = [
    ["PRON", "VERB", "NOUN", "PUNCT"],
    ["PRON", "ADV", "VERB", "NOUN", "PUNCT"],
    ["PRON", "VERB", "VERB", "NOUN", "PUNCT"],
    ["NOUN", "ADV", "ADJ", "PUNCT"],
    ["PRON", "NOUN", "VERB", "NOUN", "PUNCT"],
]


def generate_data(
    num_sentences: int,
) -> tuple[list[list[tuple[str, str]]], list[list[str]]]:
    """Generate training and test data."""
    training = []
    test = []

    for _ in range(num_sentences):
        pattern = random.choice(PATTERNS)
        sentence = []
        for tag in pattern:
            if tag in VOCAB_BY_TAG:
                word = random.choice(VOCAB_BY_TAG[tag])
                sentence.append((word, tag))
        training.append(sentence)

        # Generate test sentence (just words)
        test_pattern = random.choice(PATTERNS)
        test_sent = [random.choice(VOCAB_BY_TAG.get(t, ["嘢"])) for t in test_pattern]
        test.append(test_sent)

    return training, test


def benchmark_at_size(
    size: int,
    rustling_cls: type | None,
    pycantonese_cls: type | None,
    iterations: int = 3,
) -> ScalingPoint:
    """Run benchmark at a specific data size."""
    training, test = generate_data(num_sentences=size)

    rustling_train_time = None
    rustling_tag_time = None
    pycantonese_train_time = None
    pycantonese_tag_time = None

    for name, cls in [("rustling", rustling_cls), ("pycantonese", pycantonese_cls)]:
        if cls is None:
            continue

        # Benchmark training
        train_times = []
        for _ in range(iterations):
            gc.collect()
            model = cls(frequency_threshold=5, ambiguity_threshold=0.9, n_iter=3)
            start = time.perf_counter()
            model.train(training)
            train_times.append(time.perf_counter() - start)

        avg_train_time = sum(train_times) / len(train_times)

        # Train a model for tagging benchmark
        model = cls(frequency_threshold=5, ambiguity_threshold=0.9, n_iter=3)
        model.train(training)

        # Benchmark tagging
        tag_times = []
        for _ in range(iterations):
            gc.collect()
            start = time.perf_counter()
            for sent in test:
                model.tag(sent)
            tag_times.append(time.perf_counter() - start)

        avg_tag_time = sum(tag_times) / len(tag_times)

        if name == "rustling":
            rustling_train_time = avg_train_time
            rustling_tag_time = avg_tag_time
        else:
            pycantonese_train_time = avg_train_time
            pycantonese_tag_time = avg_tag_time

    train_speedup = None
    tag_speedup = None
    if rustling_train_time and pycantonese_train_time:
        train_speedup = pycantonese_train_time / rustling_train_time
    if rustling_tag_time and pycantonese_tag_time:
        tag_speedup = pycantonese_tag_time / rustling_tag_time

    return ScalingPoint(
        size=size,
        rustling_train_time=rustling_train_time,
        rustling_tag_time=rustling_tag_time,
        pycantonese_train_time=pycantonese_train_time,
        pycantonese_tag_time=pycantonese_tag_time,
        train_speedup=train_speedup,
        tag_speedup=tag_speedup,
    )


def print_results_table(results: list[ScalingPoint]) -> None:
    """Print results as a table."""
    print("\n" + "=" * 100)
    print("Scaling Results")
    print("=" * 100)

    # Header
    print(
        f"{'Size':<10} "
        f"{'Rustling':<12} {'Rustling':<12} "
        f"{'Pycantonese':<14} {'Pycantonese':<14} "
        f"{'Train':<10} {'Tag':<10}"
    )
    print(
        f"{'':<10} "
        f"{'Train (s)':<12} {'Tag (s)':<12} "
        f"{'Train (s)':<14} {'Tag (s)':<14} "
        f"{'Speedup':<10} {'Speedup':<10}"
    )
    print("-" * 100)

    for r in results:
        rustling_train = (
            f"{r.rustling_train_time:.4f}" if r.rustling_train_time else "N/A"
        )
        rustling_tag = f"{r.rustling_tag_time:.4f}" if r.rustling_tag_time else "N/A"
        pycantonese_train = (
            f"{r.pycantonese_train_time:.4f}" if r.pycantonese_train_time else "N/A"
        )
        pycantonese_tag = (
            f"{r.pycantonese_tag_time:.4f}" if r.pycantonese_tag_time else "N/A"
        )
        train_speedup = f"{r.train_speedup:.2f}x" if r.train_speedup else "N/A"
        tag_speedup = f"{r.tag_speedup:.2f}x" if r.tag_speedup else "N/A"

        print(
            f"{r.size:<10} "
            f"{rustling_train:<12} {rustling_tag:<12} "
            f"{pycantonese_train:<14} {pycantonese_tag:<14} "
            f"{train_speedup:<10} {tag_speedup:<10}"
        )


def print_ascii_chart(results: list[ScalingPoint], metric: str = "tag") -> None:
    """Print an ASCII bar chart of speedups."""
    print(f"\n{'=' * 60}")
    print(f"Speedup Chart ({metric.title()})")
    print("=" * 60)

    max_speedup = 0
    for r in results:
        speedup = r.train_speedup if metric == "train" else r.tag_speedup
        if speedup and speedup > max_speedup:
            max_speedup = speedup

    if max_speedup == 0:
        print("No data available")
        return

    chart_width = 40

    for r in results:
        speedup = r.train_speedup if metric == "train" else r.tag_speedup
        if speedup:
            bar_len = int((speedup / max_speedup) * chart_width)
            bar = "█" * bar_len
            print(f"{r.size:>6} | {bar} {speedup:.2f}x")
        else:
            print(f"{r.size:>6} | N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling benchmark for POS taggers")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate ASCII charts",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="500,1000,2000,5000,10000",
        help="Comma-separated list of sizes to benchmark",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Try to import implementations
    rustling_cls = None
    pycantonese_cls = None

    try:
        from rustling.taggers import AveragedPerceptronTagger

        rustling_cls = AveragedPerceptronTagger
        print("✓ rustling.taggers.AveragedPerceptronTagger available")
    except ImportError:
        print("✗ rustling not available")

    try:
        from pycantonese.pos_tagging.tagger import POSTagger

        pycantonese_cls = POSTagger
        print("✓ pycantonese.pos_tagging.POSTagger available")
    except ImportError:
        print("✗ pycantonese not available")

    if not rustling_cls and not pycantonese_cls:
        print("\nError: Neither implementation available!")
        return

    print(f"\nRunning scaling benchmark with sizes: {sizes}")

    results = []
    for size in sizes:
        print(f"\nBenchmarking size={size}...", end=" ", flush=True)
        result = benchmark_at_size(size, rustling_cls, pycantonese_cls)
        results.append(result)
        print("done")

    print_results_table(results)

    if args.plot:
        print_ascii_chart(results, "train")
        print_ascii_chart(results, "tag")


if __name__ == "__main__":
    main()
