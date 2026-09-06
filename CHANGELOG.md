# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CHAT parsing and validation are now powered by the official TalkBank
  [`chatter`](https://github.com/TalkBank/chatter) crates (`talkbank-model` +
  `talkbank-parser`). rustling's own hand-written CHAT parser and validator are
  gone. The Python API is unchanged.
- `CHAT.from_strs` no longer performs file-level validation, even with
  `strict=True`; it reports mor/word misalignment only.
  This is because `chatter`'s
  validation is whole-file (it requires a complete preamble with `@Languages`
  and `@ID`), which the bare utterance fragments `from_strs` is designed to
  accept can never satisfy. Use `CHAT.from_files` / `CHAT.from_dir` with
  `strict=True` to validate complete transcripts.
- `Participant.language` keeps every code declared in a multi-language `@ID`
  field (e.g. `"yue,eng"`) instead of only the first.
- `CHAT.from_url` pointed at a single transcript now reads it as the whole file
  it is rather than as a possible fragment, so `strict=True` validates it the
  way `CHAT.from_files` would, and its `file_path` is the transcript's name
  from the URL rather than a generated UUID. The same parity applies to
  `CHAT.from_zip`, where the entry name is the transcript's name: both now
  report the missing-envelope and `@Media`-name rules they used to skip.
- Strict-mode error messages now carry `chatter`'s canonical error code
  alongside the variant name (e.g. `[E305 MissingTerminator]` where they
  previously read `[MissingTerminator]`). The code is the identifier
  `chatter`'s error specs and documentation are keyed by, and it is stable
  across the enum renames the variant names have seen.
- **BREAKING:** `strict=True` now reports every problem it found, not the first
  one. The `ValueError` opens with `Found N problem(s) in M file(s):` and lists
  them, capped at 50 with a count of what was omitted. `chatter`'s diagnostics
  and rustling's mor/word misalignments are listed together rather than in
  sequence: the alignment check used to run first and raise on its own, so an
  utterance that broke a `chatter` rule *and* came out misaligned was rejected
  under rustling's generic count message and the rule actually broken was never
  named. Code matching on the exact text of these errors will need updating.
- rustling now hands `chatter` the source as written. It previously trimmed
  every line, dropped blank lines and folded continuation lines into the line
  above -- reasonable when rustling parsed CHAT itself, but each transformation
  removed something `chatter` needs to see. Only two normalizations remain, and
  both only remove noise: a leading byte-order mark, and the indent shared by
  every line of the file (so a source indented as a whole still parses).
  `raw_lines`, and therefore `to_strs` / `to_files` output, is correspondingly
  closer to the input than before.

### Added

- `CHAT.diagnostics` returns the `Diagnostic` objects `chatter` reported for
  the loaded files, each carrying `code`, `name`, `is_error`, `message` and
  `file_path`. Diagnostics are collected whether or not the data was loaded
  with `strict=True`, so choosing to load a transcript leniently no longer
  means giving up the ability to find out what is wrong with it. Semantic
  validation still only runs under `strict=True`, so a lenient load reports
  parse-level diagnostics only.

### Fixed

- `to_chat` from SRT, ELAN and TextGrid no longer loses words. Those converters
  emit CHAT whose utterance bodies are source text -- a subtitle line, an
  annotation value -- and then read that text back with a CHAT grammar, which
  treats punctuation as structure and drops whatever it cannot account for:
  `Hello [world] and <stuff> & more!` came back as `more !`, `50% off` as
  `off`, and `cost $5` as `cost`. They now build the CHAT objects directly from
  the generated text, as the CoNLL-U converter already did. Words are the
  whitespace-separated runs of the source, with a trailing `.`, `?` or `!` on
  the last word kept as its own terminator token; the emitted `.cha` text is
  unchanged.
- `Utterance.audible` no longer leaks a stray `[` for a code-switch span.
  `<como estas> [@s] there .` came back as `como estas [ there .`, because
  `[@s]` and `[@s:code]` were not among the annotations the audible view knows
  to drop and so fell through to the branch that keeps an unrecognized bracket
  as a word. The span marks the language of the words it scopes; those words
  are speech and stay, the marker does not.
- A continuation line beginning with `@`, `*` or `%` is no longer promoted to a
  line of its own. CHAT wraps a long header or tier by starting the next line
  with a tab, but rustling classified lines by their first character *after*
  trimming, which discards exactly the indentation that marks a continuation.
  A wrapped `@Comment` listing header names (`@Transcriber, @Transcription` on
  the continuation line) was therefore split off as a bare line, which
  `chatter` then reported as unparsable file-level content, so a valid
  transcript failed to load under `strict=True`.
- Blank lines now reach `chatter`, which is what E747 (`BlankLineNotAllowed`)
  exists to report. rustling stripped them before `chatter` could see them, so
  a rule `chatter` implements could never fire.
- `Utterance.annotated`, `Utterance.audible` and the raw text in
  `Utterance.tiers` are no longer empty in wasm/Pyodide builds. Those builds
  use `chatter`'s re2c parser, which records no byte offsets at all, so slicing
  the source with its spans yielded nothing for every utterance in every file.
  rustling now falls back to its own line-block reading of the source when the
  parser reports no spans, so these attributes are populated on both backends.

### Known limitations

- Preclitics (`$` on the `%mor` tier, e.g. `v|da-give$pro|me&dat-me`) are not
  supported: `chatter` does not model them, so such a `%mor` tier fails to
  parse and the utterance is reported as a mor/word misalignment with no
  `pos`/`mor` on its tokens. The raw tier text is still available from
  `Utterance.tiers`. Postclitics (`~`) are unaffected.
- One check the previous hand-written validator performed has no `chatter`
  equivalent, so `strict=True` no longer flags it: a tone terminator
  (a trailing `-.`, `-?` or `-,.`). Validation is stricter overall (see
  above), but this specific check was lost.
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
