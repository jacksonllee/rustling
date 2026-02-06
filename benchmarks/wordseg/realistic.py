#!/usr/bin/env python
"""Benchmark with real-world text data.

This script benchmarks word segmentation using realistic text data,
simulating scenarios like Chinese word segmentation or similar
unsegmented language processing.

Usage:
    python benchmarks/realistic.py
    python benchmarks/realistic.py --lang chinese
    python benchmarks/realistic.py --lang english
"""

from __future__ import annotations

import argparse
import gc
import time
import random
from typing import Any

# Sample text data for different "languages" / use cases
# In real Chinese text, there are no spaces - perfect for word segmentation
SAMPLE_CHINESE_WORDS = [
    "我",
    "你",
    "他",
    "她",
    "它",
    "我們",
    "你們",
    "他們",
    "是",
    "不是",
    "有",
    "沒有",
    "在",
    "不在",
    "會",
    "不會",
    "這",
    "那",
    "這個",
    "那個",
    "這裡",
    "那裡",
    "這些",
    "那些",
    "很",
    "非常",
    "特別",
    "相當",
    "極其",
    "十分",
    "好",
    "壞",
    "大",
    "小",
    "多",
    "少",
    "高",
    "低",
    "今天",
    "明天",
    "昨天",
    "現在",
    "以前",
    "以後",
    "學習",
    "工作",
    "生活",
    "研究",
    "發展",
    "進步",
    "中國",
    "美國",
    "日本",
    "英國",
    "法國",
    "德國",
    "北京",
    "上海",
    "廣州",
    "深圳",
    "香港",
    "台北",
    "電腦",
    "手機",
    "網路",
    "軟體",
    "程式",
    "程式碼",
    "人工智慧",
    "機器學習",
    "深度學習",
    "自然語言處理",
    "大數據",
    "雲端運算",
    "網際網路",
    "物聯網",
]

SAMPLE_ENGLISH_WORDS = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "word",
    "segmentation",
    "algorithm",
    "implementation",
    "performance",
    "benchmark",
    "processing",
    "language",
    "natural",
    "computational",
    "linguistics",
    "research",
]


def generate_corpus(
    words: list[str],
    num_sentences: int = 1000,
    min_words: int = 5,
    max_words: int = 20,
) -> tuple[list[tuple[str, ...]], list[str]]:
    """Generate training and test data from a word list.

    Returns:
        Tuple of (training_sentences, test_sentences_unsegmented)
    """
    training = []
    test_unsegmented = []

    for _ in range(num_sentences):
        n_words = random.randint(min_words, max_words)
        sentence_words = random.choices(words, k=n_words)
        training.append(tuple(sentence_words))
        test_unsegmented.append("".join(sentence_words))

    return training, test_unsegmented


def benchmark_implementation(
    cls: type,
    training: list[tuple[str, ...]],
    test: list[str],
    max_word_length: int,
    iterations: int = 5,
) -> dict[str, Any]:
    """Benchmark a single implementation.

    Returns:
        Dictionary with timing results.
    """
    # Training time
    model = cls(max_word_length=max_word_length)
    gc.collect()

    train_times = []
    for _ in range(iterations):
        model = cls(max_word_length=max_word_length)
        start = time.perf_counter()
        model.fit(training)
        train_times.append(time.perf_counter() - start)

    # Prediction time
    model = cls(max_word_length=max_word_length)
    model.fit(training)
    gc.collect()

    predict_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = model.predict(test)
        # Force evaluation of lazy iterators (wordseg returns map objects)
        list(result)
        predict_times.append(time.perf_counter() - start)

    return {
        "train_time_avg": sum(train_times) / len(train_times),
        "train_time_min": min(train_times),
        "predict_time_avg": sum(predict_times) / len(predict_times),
        "predict_time_min": min(predict_times),
        "total_time_avg": sum(train_times) / len(train_times)
        + sum(predict_times) / len(predict_times),
    }


def run_realistic_benchmark(
    lang: str = "chinese",
    num_sentences: int = 1000,
    iterations: int = 5,
) -> None:
    """Run benchmark with realistic data."""
    # Choose word list
    if lang == "chinese":
        words = SAMPLE_CHINESE_WORDS
        max_word_length = 6  # Chinese words are typically shorter in chars
    else:
        words = SAMPLE_ENGLISH_WORDS
        max_word_length = 15

    print(f"\n{'='*60}")
    print(f"REALISTIC BENCHMARK: {lang.upper()} text simulation")
    print(f"{'='*60}")
    print(f"Vocabulary size: {len(words)} words")
    print(f"Sentences: {num_sentences}")
    print(f"Iterations: {iterations}")

    # Generate data
    random.seed(42)
    training, test = generate_corpus(words, num_sentences=num_sentences)

    avg_test_len = sum(len(s) for s in test) / len(test)
    print(f"Avg test sentence length: {avg_test_len:.1f} chars")

    # Import implementations
    rustling_cls = None
    wordseg_cls = None

    try:
        from rustling.wordseg import LongestStringMatching as RustlingLSM

        rustling_cls = RustlingLSM
        print("\n✓ rustling available")
    except ImportError as e:
        print(f"\n✗ rustling not available: {e}")

    try:
        from wordseg import LongestStringMatching as WordsegLSM

        wordseg_cls = WordsegLSM
        print("✓ wordseg (Python) available")
    except ImportError as e:
        print(f"✗ wordseg not available: {e}")

    if rustling_cls is None and wordseg_cls is None:
        print("\nNo implementations available!")
        return

    print(f"\n📊 LongestStringMatching (max_word_length={max_word_length})")
    print("-" * 50)

    results = {}

    if rustling_cls:
        results["rustling"] = benchmark_implementation(
            rustling_cls, training, test, max_word_length, iterations
        )
        r = results["rustling"]
        train_avg = r["train_time_avg"] * 1000
        train_min = r["train_time_min"] * 1000
        pred_avg = r["predict_time_avg"] * 1000
        pred_min = r["predict_time_min"] * 1000
        throughput = num_sentences / r["predict_time_avg"]
        print("\nrustling (Rust):")
        print(f"  Training:   {train_avg:.2f}ms (min: {train_min:.2f}ms)")
        print(f"  Prediction: {pred_avg:.2f}ms (min: {pred_min:.2f}ms)")
        print(f"  Throughput: {throughput:,.0f} sentences/sec")

    if wordseg_cls:
        results["wordseg"] = benchmark_implementation(
            wordseg_cls, training, test, max_word_length, iterations
        )
        r = results["wordseg"]
        train_avg = r["train_time_avg"] * 1000
        train_min = r["train_time_min"] * 1000
        pred_avg = r["predict_time_avg"] * 1000
        pred_min = r["predict_time_min"] * 1000
        throughput = num_sentences / r["predict_time_avg"]
        print("\nwordseg (Python):")
        print(f"  Training:   {train_avg:.2f}ms (min: {train_min:.2f}ms)")
        print(f"  Prediction: {pred_avg:.2f}ms (min: {pred_min:.2f}ms)")
        print(f"  Throughput: {throughput:,.0f} sentences/sec")

    # Comparison
    if rustling_cls and wordseg_cls:
        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")

        train_speedup = (
            results["wordseg"]["train_time_avg"] / results["rustling"]["train_time_avg"]
        )
        predict_speedup = (
            results["wordseg"]["predict_time_avg"]
            / results["rustling"]["predict_time_avg"]
        )
        total_speedup = (
            results["wordseg"]["total_time_avg"] / results["rustling"]["total_time_avg"]
        )

        print(f"\n⚡ Training speedup:   {train_speedup:.1f}x faster")
        print(f"⚡ Prediction speedup: {predict_speedup:.1f}x faster")
        print(f"⚡ Overall speedup:    {total_speedup:.1f}x faster")

        if predict_speedup > 1:
            msg = f"\n🚀 rustling processes {predict_speedup:.1f}x more sentences"
            print(msg + " per second!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark with realistic text data")
    parser.add_argument(
        "--lang",
        choices=["chinese", "english"],
        default="chinese",
        help="Language/text type to simulate",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=1000,
        help="Number of sentences",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )

    args = parser.parse_args()

    run_realistic_benchmark(
        lang=args.lang,
        num_sentences=args.sentences,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
