# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CHAT parsing and validation are now powered by the official TalkBank
  [`chatter`](https://github.com/TalkBank/chatter) crates (`talkbank-model` +
  `talkbank-parser`) instead of rustling's own hand-written parser. The Python
  API is unchanged. Because `chatter` is not yet published to crates.io, it is
  pinned as a git dependency (tag `v0.5.1`).
  Since crates.io doesn't allow git dependencies, this branch using `chatter`
  won't be merged in the Rustling's `main` branch for release yet --
  we need to wait for `chatter`'s crates to be available on crates.io.
- CHAT validation now follows the official CLAN-CHECK-parity rules from
  `chatter`, so `strict=True` file loading flags more issues than before and
  error messages differ.
- `CHAT.from_strs` no longer performs file-level validation, even with
  `strict=True`; it reports mor/word misalignment only. This is a change from
  0.9.0, where string input was validated by rustling's own rules. `chatter`'s
  validation is whole-file (it requires a complete preamble with `@Languages`
  and `@ID`), which the bare utterance fragments `from_strs` is designed to
  accept can never satisfy. Use `CHAT.from_files` / `CHAT.from_dir` with
  `strict=True` to validate complete transcripts.
- Word-to-`%mor` alignment now follows `chatter`'s own alignment rules, so the
  words rustling counts are exactly the words `chatter` expects a `%mor` item
  for. This affects utterances containing `[e]`-excluded material, phonological
  or sign groups (`‹…›`), and CA separators such as `;` and `:`.
- `Participant.language` keeps every code declared in a multi-language `@ID`
  field (e.g. `"yue,eng"`) instead of only the first.

### Known limitations

- Preclitics (`$` on the `%mor` tier, e.g. `v|da-give$pro|me&dat-me`) are not
  supported: `chatter` does not model them, so such a `%mor` tier fails to
  parse and the utterance is reported as a mor/word misalignment with no
  `pos`/`mor` on its tokens. The raw tier text is still available from
  `Utterance.tiers`. Postclitics (`~`) are unaffected.
- Some checks the previous hand-written validator performed have no `chatter`
  equivalent yet, so `strict=True` no longer flags them: `@Media` name vs.
  file name mismatches, tone terminators, invalid language codes, and
  zero-word forms. Validation is stricter overall (see above), but these
  specific checks were lost.
- The `[x N]` repetition marker is not yet implemented by `chatter`'s grammar,
  so `strict=True` rejects files containing it and lenient parsing degrades the
  affected utterance's word tokens (the marker's content can surface as word
  tokens and drop the rest of the utterance). No data is lost otherwise:
  `Utterance.annotated` preserves the raw main tier, and `Utterance.audible`
  still expands `[x N]` (e.g., `play [x 3]` -> `play play play`), so legacy
  transcripts loaded with `strict=False` still yield the expanded form.

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
