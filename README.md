<img src="https://raw.githubusercontent.com/jacksonllee/rustling/main/python/docs/_static/logo-with-text.svg" alt="Rustling" height="120">

[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)

Rustling is a blazingly fast library for computational linguistics.
It is written in Rust, with Python bindings.

Documentation: [Python](https://rustling.readthedocs.io/) | [Rust](https://docs.rs/rustling)

## Features

- N-grams
- Language models
- Hidden Markov model
- Word segmentation
- Part-of-speech tagging
- CHAT parsing for TalkBank and CHILDES data

## Performance

| Component | Task | Speedup | vs. |
|---|---|---|---|
| **Language Models** | Fit | **10x** | NLTK |
|  | Score | **1.9x** | NLTK |
|  | Generate | **106--114x** | NLTK |
| **Word Segmentation** | LongestStringMatching | **9x** | wordseg |
| **POS Tagging** | Training | **5x** | NLTK |
|  | Tagging | **18x** | NLTK |
| **HMM** | Fit | **13x** | hmmlearn |
|  | Predict | **0.9x** | hmmlearn |
|  | Score | **5x** | hmmlearn |
| **CHAT Parsing** | Reading from a ZIP archive | **43x** | pylangacq |
|  | Reading from strings | **70x** | pylangacq |
|  | Parsing utterances | **15x** | pylangacq |
|  | Parsing tokens | **9x** | pylangacq |

See [`benchmarks/`](https://github.com/jacksonllee/rustling/tree/main/benchmarks) for reproduction scripts.


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
