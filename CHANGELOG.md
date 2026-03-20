# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
