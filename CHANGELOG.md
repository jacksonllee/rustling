# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-11-20

This release is available from PyPI and GitHub, but not from crates.io:
`chatter` is not yet on crates.io, and crates.io does not accept a crate with
git dependencies. Rust users can depend on the release tag instead:
`rustling = { git = "https://github.com/jacksonllee/rustling", tag = "v0.10.0" }`.
The Rust API documentation on docs.rs remains at v0.9.0; run `cargo doc --open`
in a project depending on rustling for the current version.

### Changed

- CHAT parsing and validation are now powered by the official TalkBank
  [`chatter`](https://github.com/TalkBank/chatter) crates, replacing rustling's
  own hand-written CHAT parser and validator. The Python API is unchanged.
- **BREAKING:** `strict=True` now uses `chatter`'s validation, which is
  stricter. Files that loaded before may be rejected, e.g., for a missing
  `@UTF8` / `@Begin` / `@End`, a blank line, or a header with nothing after its
  colon (such as an empty `@Comment:`).
- **BREAKING:** `strict=True` errors list every problem found rather than only
  the first, each tagged with `chatter`'s error code
  (e.g., `[E305 MissingTerminator]`).
- **BREAKING:** `CHAT.from_strs` with `strict=True` checks only mor/word
  alignment, because `chatter` validates whole files only. Use
  `CHAT.from_files` or `CHAT.from_dir` to validate complete transcripts.
- `to_strs` and `to_files` now write the source as it was read; continuation
  lines are no longer joined onto the line above.

### Added

- `CHAT.diagnostics` returns the problems `chatter` reported for the loaded
  files. It is available after a `strict=False` load too, though without
  `strict=True` it covers parsing problems only.
- Support for Python 3.15.

### Removed

- Dropped support for Python 3.10.

### Fixed

- `HiddenMarkovModel.fit` without labels (Baum-Welch EM) no longer corrupts
  the model when a state has no expected transitions or emissions in the
  training data, e.g., when training only on single-token sequences. Such
  rows previously became NaN, after which every multi-token sequence scored
  `-inf`; they now keep their existing values.
  ([#36](https://github.com/jacksonllee/rustling/pull/36), thanks to
  [@antonsoo](https://github.com/antonsoo))

### Known limitations

- The `[x N]` repetition marker is not supported by `chatter`:
  `strict=True` rejects files containing it, and with `strict=False` the
  affected utterance's tokens can be wrong. `Utterance.audible` still expands
  it (e.g., `play [x 3]` -> `play play play`).

## [0.9.0] - 2026-07-20

### Added

- CHAT data: New attribute `Utterance.annotated` for the main tier transcription
  with all TalkBank annotations preserved

## [0.8.0] - 2026-03-20

### Added

- Support for data formats:
   * CoNLL-U for Universal Dependencies
   * ELAN for annotated multimedia data
   * TextGrid for Praat annotations
   * SRT for subtitles
- CHAT data handling:
   * Added a convenience function `read_chat.
   * Added `from_git` and `from_url` methods for remote data sources.

## [0.7.0] - 2026-03-14

### Added

- Word segmentation:
   * Added `score` method for the HMM and DAG-HMM segmenters.
   * `predict` method can optionally output offsets for the (start, end) indices
     of segmented words compared to the original string.
- CHAT parsing: Support custom tier names other than the standard %mor and %gra.
- Python model classes are now subclassable.

### Changed

- Ngram counters: `Ngrams.most_common` now sorts tuples lexicographically
  when counts are tied.
- CHAT parsing:
   * If a date is available at `Headers`'s `date`,
     it's now a Python `datetime.date` object instead of a string.
   * In handling the main tier transcription for creating `Token` objects:
      - Special form markers suffixed with "@" are now stripped.
      - Words that have partiallly parenthetical material have the parentheses
        removed, e.g., (un)til -> until, sit(ting) -> sitting.
   * Renamed the `CHAT.raw` attribute to `CHAT.audible` for a best-effort,
     audibly faithful transcription string, to facilitate automatic speech recognition,
     forced alignment, etc.
   * A subset of the testchat/bad dataset is now used to validate CHAT data format.
- Refactored core Rust code so that Rust-only consumers no longer need PyO3/Python.

## [0.6.0] - 2026-03-05

### Added

- Hidden Markov Model (HMM)
- Word segmentation: Added DAG-HMM word segmenter
- CHAT parsing: Added `from_utterances` method

### Changed

- Models are now persisted as a zstd-compressed FlatBuffers binary.

## [0.5.0] - 2026-02-18

### Added

- CHAT parsing for TalkBank and CHILDES data

## [0.4.0] - 2026-02-08

### Added

- N-grams and language models

## [0.3.0] - 2026-02-06

### Added

- Averaged perceptron tagger

## [0.2.0] - 2026-02-04

- Initial release, with longest string matching and random segmenter for word segmentation
