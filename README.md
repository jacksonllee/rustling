# Rustling

[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)

Rustling is a blazingly fast library of tools for computational linguistics.
It is written in Rust, with Python bindings.

Documentation: [Python](https://rustling.readthedocs.io/) | [Rust](https://docs.rs/rustling)

## Features

- **Word Segmentation** — Models for segmenting unsegmented text into words
  - `LongestStringMatching` — Greedy left-to-right longest match segmenter
  - `RandomSegmenter` — Random baseline segmenter

- **Part-of-speech Tagging**
  - `AveragedPerceptronTagger` - Averaged perceptron tagger


## Performance

See [benchmarks](https://github.com/jacksonllee/rustling/tree/main/benchmarks)


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
