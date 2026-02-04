# rustling

[![Rust](https://github.com/jacksonllee/rustling/actions/workflows/rust.yml/badge.svg)](https://github.com/jacksonllee/rustling/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/rustling.svg)](https://crates.io/crates/rustling)

[![Python](https://github.com/jacksonllee/rustling/actions/workflows/python.yml/badge.svg)](https://github.com/jacksonllee/rustling/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/rustling.svg)](https://pypi.org/project/rustling/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/rustling.svg)](https://pypi.org/project/rustling/)

**rustling** is a library of tools for computational linguistics, implemented in Rust with Python bindings.

## Features

- **Word Segmentation** — Models for segmenting unsegmented text into words
  - `LongestStringMatching` — Greedy left-to-right longest match segmenter
  - `RandomSegmenter` — Random baseline segmenter

## Installation

### Python

```bash
pip install rustling
```

### Rust

```bash
cargo add rustling
```

## Usage

### Python

```python
from rustling.wordseg import LongestStringMatching, RandomSegmenter

# Longest String Matching
model = LongestStringMatching(max_word_length=4)
model.fit([
    ("this", "is", "a", "sentence"),
    ("that", "is", "not", "a", "sentence"),
])
result = model.predict(["thatisadog", "thisisnotacat"])
print(result)
# [['that', 'is', 'a', 'd', 'o', 'g'], ['this', 'is', 'not', 'a', 'c', 'a', 't']]

# Random Segmenter (no training needed)
segmenter = RandomSegmenter(prob=0.3)
result = segmenter.predict(["helloworld"])
print(result)
# e.g., [['hel', 'lo', 'wor', 'ld']] (varies due to randomness)
```

### Rust

```rust
use rustling::wordseg::{LongestStringMatching, RandomSegmenter};

fn main() {
    // Longest String Matching
    let mut model = LongestStringMatching::new(4).unwrap();
    model.fit(vec![
        vec!["this".into(), "is".into(), "a".into(), "sentence".into()],
        vec!["that".into(), "is".into(), "not".into(), "a".into(), "sentence".into()],
    ]);
    let result = model.predict(vec!["thatisadog".into(), "thisisnotacat".into()]);
    println!("{:?}", result);
    // [["that", "is", "a", "d", "o", "g"], ["this", "is", "not", "a", "c", "a", "t"]]

    // Random Segmenter (no training needed)
    let segmenter = RandomSegmenter::new(0.3).unwrap();
    let result = segmenter.predict(vec!["helloworld".into()]);
    println!("{:?}", result);
    // e.g., [["hel", "lo", "wor", "ld"]] (varies due to randomness)
}
```

## License

MIT License

## Links

* Author: [Jackson L. Lee](https://jacksonllee.com)
* Source code: <https://github.com/jacksonllee/rustling>
