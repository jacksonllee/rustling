#!/usr/bin/env python
"""Scaling benchmark to show performance at different data sizes.

This script generates a performance scaling chart data, showing how
rustling and wordseg scale with increasing data sizes.

Usage:
    python benchmarks/scaling.py
    python benchmarks/scaling.py --plot  # Generate ASCII chart
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
    rustling_time: float | None
    wordseg_time: float | None
    speedup: float | None


def generate_data(
    num_words: int, num_sentences: int
) -> tuple[list[tuple[str, ...]], list[str]]:
    """Generate training and test data."""
    import string

    # Generate vocabulary
    vocab = set()
    while len(vocab) < num_words:
        length = random.randint(2, 8)
        word = "".join(random.choices(string.ascii_lowercase, k=length))
        vocab.add(word)
    vocab_list = list(vocab)

    # Generate training sentences
    training = []
    for _ in range(num_sentences):
        sent = tuple(random.choices(vocab_list, k=random.randint(5, 15)))
        training.append(sent)

    # Generate test sentences
    test = []
    for _ in range(num_sentences):
        words = random.choices(vocab_list, k=random.randint(5, 15))
        test.append("".join(words))

    return training, test


def benchmark_at_size(
    size: int,
    rustling_cls: type | None,
    wordseg_cls: type | None,
    iterations: int = 3,
) -> ScalingPoint:
    """Run benchmark at a specific data size."""
    training, test = generate_data(num_words=size, num_sentences=size)

    rustling_time = None
    wordseg_time = None

    for name, cls in [("rustling", rustling_cls), ("wordseg", wordseg_cls)]:
        if cls is None:
            continue

        model = cls(max_word_length=10)
        model.fit(training)

        gc.collect()

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = model.predict(test)
            # Force evaluation of lazy iterators (wordseg returns map objects)
            list(result)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)

        if name == "rustling":
            rustling_time = avg_time
        else:
            wordseg_time = avg_time

    speedup = None
    if rustling_time and wordseg_time:
        speedup = wordseg_time / rustling_time

    return ScalingPoint(
        size=size,
        rustling_time=rustling_time,
        wordseg_time=wordseg_time,
        speedup=speedup,
    )


def draw_ascii_chart(points: list[ScalingPoint]) -> None:
    """Draw an ASCII chart of the results."""
    if not points:
        return

    # Get max speedup for scaling
    speedups = [p.speedup for p in points if p.speedup]
    if not speedups:
        print("No comparison data available for chart.")
        return

    max_speedup = max(speedups)
    chart_width = 50

    print("\n" + "=" * 60)
    print("SPEEDUP BY DATA SIZE (rustling vs wordseg)")
    print("=" * 60)
    print()

    for point in points:
        if point.speedup is None:
            continue

        bar_len = int((point.speedup / max_speedup) * chart_width)
        bar = "█" * bar_len

        print(f"{point.size:>6} sentences │ {bar} {point.speedup:.1f}x")

    print()
    print(f"{'':>6}           └{'─' * chart_width}─")
    print(f"{'':>6}            0x{' ' * (chart_width//2 - 2)}{max_speedup:.0f}x")


def run_scaling_benchmark(
    sizes: list[int] | None = None,
    show_plot: bool = False,
) -> list[ScalingPoint]:
    """Run scaling benchmark across multiple sizes."""
    if sizes is None:
        sizes = [100, 250, 500, 1000, 2000, 5000]

    # Import implementations
    rustling_cls = None
    wordseg_cls = None

    try:
        from rustling.wordseg import LongestStringMatching as RustlingLSM

        rustling_cls = RustlingLSM
        print("✓ rustling available")
    except ImportError as e:
        print(f"✗ rustling not available: {e}")

    try:
        from wordseg import LongestStringMatching as WordsegLSM

        wordseg_cls = WordsegLSM
        print("✓ wordseg available")
    except ImportError as e:
        print(f"✗ wordseg not available: {e}")

    if rustling_cls is None and wordseg_cls is None:
        print("No implementations available!")
        return []

    print("\n" + "=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)

    random.seed(42)
    points = []

    print(
        "\n{:>10} {:>12} {:>12} {:>10}".format("Size", "rustling", "wordseg", "Speedup")
    )
    print("-" * 50)

    for size in sizes:
        point = benchmark_at_size(size, rustling_cls, wordseg_cls)
        points.append(point)

        rustling_str = (
            f"{point.rustling_time*1000:.1f}ms" if point.rustling_time else "N/A"
        )
        wordseg_str = (
            f"{point.wordseg_time*1000:.1f}ms" if point.wordseg_time else "N/A"
        )
        speedup_str = f"{point.speedup:.1f}x" if point.speedup else "N/A"

        print(f"{size:>10} {rustling_str:>12} {wordseg_str:>12} {speedup_str:>10}")

    if show_plot:
        draw_ascii_chart(points)

    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scaling with data size")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show ASCII chart of results",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        help="Comma-separated list of sizes to test (e.g., '100,500,1000')",
    )

    args = parser.parse_args()

    sizes = None
    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]

    run_scaling_benchmark(sizes=sizes, show_plot=args.plot)


if __name__ == "__main__":
    main()
