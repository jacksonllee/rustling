<div align="center">
  <a href="https://github.com/jacksonllee/rustling">
    <img src="https://raw.githubusercontent.com/jacksonllee/rustling/main/python/docs/_static/logo-with-text.svg" alt="Rustling" height="120">
  </a>
</div>
<br>

[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/rustling.svg)](https://anaconda.org/conda-forge/rustling)

Rustling is a blazingly fast library for computational linguistics.

Documentation: [Python](https://docs.rustling.io/) | [Rust](https://docs.rs/rustling)

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
| **Language Models** | Fit | **11x** | NLTK |
|  | Score | **2x** | NLTK |
|  | Generate | **86--107x** | NLTK |
| **Word Segmentation** | LongestStringMatching | **9x** | wordseg |
| **POS Tagging** | Training | **5x** | NLTK |
|  | Tagging | **17x** | NLTK |
| **HMM** | Fit | **14x** | hmmlearn |
|  | Predict | **0.9x** | hmmlearn |
|  | Score | **5x** | hmmlearn |
| **CHAT Parsing** | Reading from a ZIP archive | **30x** | pylangacq |
|  | Reading from strings | **35x** | pylangacq |
|  | Parsing utterances | **15x** | pylangacq |
|  | Parsing tokens | **8x** | pylangacq |

See [`benchmarks/`](https://github.com/jacksonllee/rustling/tree/main/benchmarks) for reproduction scripts.


## Installation

### Python

Using pip:

```bash
pip install rustling
```

Using conda:

```bash
conda install -c conda-forge rustling
```

For Pyodide, pre-built WASM wheels (with multithreading disabled, as Pyodide does not support it)
are available from each [GitHub release](https://github.com/jacksonllee/rustling/releases)
— look for the ``.whl`` file with ``emscripten`` in the filename.

### Rust

```bash
cargo add rustling
```

## License

MIT License
