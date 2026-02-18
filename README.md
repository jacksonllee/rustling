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

- **CHAT Parsing** — Parser for CHAT transcription files (CHILDES/TalkBank)
  - `CHAT` — Read and query CHAT data from directories, files, strings, or ZIP archives


## Performance

Benchmarked against pure Python implementations from NLTK, wordseg (v0.0.5), and pylangacq (v0.19.1).
See [`benchmarks/`](benchmarks/) for full details and reproduction scripts.

| Component | Task | Speedup | vs. |
|-----------|------|---------|-----|
| **Language Models** | Fit | **10x** | NLTK |
| | Score | **2x** | NLTK |
| | Generate | **80–112x** | NLTK |
| **Word Segmentation** | LongestStringMatching | **9x** | wordseg |
| | RandomSegmenter | **1.1x** | wordseg |
| **POS Tagging** | Training | **5x** | NLTK |
| | Tagging | **7x** | NLTK |
| **CHAT Parsing** | from_dir | **55x** | pylangacq |
| | from_zip | **48x** | pylangacq |
| | from_files | **63x** | pylangacq |
| | from_strs | **116x** | pylangacq |
| | words() | **3x** | pylangacq |
| | utterances() | **15x** | pylangacq |


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
