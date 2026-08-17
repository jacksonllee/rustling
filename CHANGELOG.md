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
  pinned as a git dependency (tag `v0.12.0`).
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
- `strict=True` file loading now reports E531, which requires the `@Media`
  header's filename to match the transcript's own name: `foo.cha` carrying
  `@Media: bar, audio` is rejected. rustling previously validated without
  telling `chatter` what the transcript was called, which switched off every
  rule about a transcript's own name. `CHAT.from_strs` is unaffected -- string
  input has no file name, so those rules correctly do not run there.
- `strict=True` file loading rejects three kinds of transcript that `chatter`
  used to accept in silence, because `chatter` v0.12.0 made the corresponding
  rules report. A corpus that loaded cleanly under the previous pin may not
  load now.
  - E241 covers the illegal spellings of the untranscribed markers. `www`,
    `xxx` and `yyy` are the canonical forms, and the wrong spellings are now
    derived from them rather than listed, so `ww` is caught alongside the `xx`
    and `yy` that already were, as are miscased forms such as `Www` and `XX`.
  - E756 covers every dependent tier whose body is free text, not only `%x…`
    ones. A tier line with an empty or whitespace-only payload declares
    nothing, so `%eng:` and `%tim:` with no content are rejected the way
    `%xtst:` already was.
  - A file declaring an empty `@Participants` set is no longer exempt from the
    participants check. The empty declaration used to switch the check off, so
    the files least likely to be well formed were the ones that escaped it;
    speakers such a file goes on to use are now reported (E522).
- `Utterance.audible` drops `ww` as well as `xx` and `yy`. All three are
  illegal short spellings of the untranscribed markers rather than transcribed
  words, and `ww` was missing from the list for the same reason `chatter` used
  to miss it: the spellings were enumerated instead of derived. `www`, `xxx`
  and `yyy` are still kept -- that material was audible.
- Word-to-`%mor` alignment now follows `chatter`'s own alignment rules, so the
  words rustling counts are exactly the words `chatter` expects a `%mor` item
  for. This affects utterances containing `[e]`-excluded material, phonological
  or sign groups (`‹…›`), and CA separators such as `;` and `:`.
- `Participant.language` keeps every code declared in a multi-language `@ID`
  field (e.g. `"yue,eng"`) instead of only the first.
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
- Some checks the previous hand-written validator performed have no `chatter`
  equivalent yet, so `strict=True` no longer flags them: tone terminators,
  invalid language codes, and zero-word forms. Validation is stricter overall
  (see above), but these specific checks were lost. `@Media` name vs. file name
  mismatches were on this list too until `chatter` v0.11.0; they are checked
  again, as E531.
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
