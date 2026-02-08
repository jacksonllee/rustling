# Rustling

[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)

Rustling is a blazingly fast library for computational linguistics.
It is written in Rust, with Python bindings.

Documentation: [Python](https://rustling.readthedocs.io/) | [Rust](https://docs.rs/rustling)

## Features

- **Language Models** — N-gram language models with smoothing
  - `MLE` — Maximum Likelihood Estimation (no smoothing)
  - `Lidstone` — Lidstone (additive) smoothing
  - `Laplace` — Laplace (add-one) smoothing

- **Word Segmentation** — Models for segmenting unsegmented text into words
  - `LongestStringMatching` — Greedy left-to-right longest match segmenter
  - `RandomSegmenter` — Random baseline segmenter

- **Part-of-speech Tagging**
  - `AveragedPerceptronTagger` - Averaged perceptron tagger


## Performance

Benchmarked against pure Python implementataions from NLTK and wordseg.
See [`benchmarks/`](benchmarks/) for full details and reproduction scripts.

| Component | Task | Speedup | vs. |
|-----------|------|---------|-----|
| **Language Models** | Fit | **11x** | NLTK |
| | Score | **2x** | NLTK |
| | Generate | **25–39x** | NLTK |
| **Word Segmentation** | LongestStringMatching | **14x** | wordseg |
| | RandomSegmenter | **12x** | wordseg |
| **POS Tagging** | Training | **5x** | NLTK |
| | Tagging | **6x** | NLTK |


## Installation

### Python

```bash
pip install rustling
```

### Rust

```bash
cargo add rustling
```

## License

MIT License
