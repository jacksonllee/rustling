//! Adapter from TalkBank `chatter`'s typed CHAT model into rustling's CHAT
//! data structures.
//!
//! CHAT parsing is delegated to the official `chatter` crates
//! (`talkbank-model` + a target-specific parser backend); this module maps the
//! resulting typed [`talkbank_model::ChatFile`] into rustling's existing
//! [`Headers`]/[`Utterance`]/[`Token`] structures so the Python API is
//! unchanged.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::chat::header::{Age, ChangeableHeader, Headers, Media, Participant};
use crate::chat::reader::{MisalignmentCounts, MisalignmentInfo, MorItem, build_tokens};
use crate::chat::utterance::{Gra, Utterance};

use talkbank_model::model::TranscriptName;
use talkbank_model::{
    AgeValue, ChatFile as TbChatFile, DependentTier, ErrorCode, ErrorCollector, GraTier, Header,
    Line, MorTier, MorWord, ParseError, ReplacedWord, Separator, Severity, TierDomain,
    UtteranceContent, Word, WriteChat,
};
// chatter's canonical alignment policy: the same traversal and word-inclusion
// rules its own `%mor` validation uses, so rustling's counts cannot drift.
use talkbank_model::alignment::helpers::{
    WordItem, annotations_have_alignment_ignore, counts_for_tier, is_tag_marker_separator,
    walk_words,
};

// ---------------------------------------------------------------------------
// Parser backend (target-gated)
// ---------------------------------------------------------------------------
//
// Both parsers implement chatter's `ChatParser` trait (as of chatter v0.4.0),
// but the backend still has to be cfg-selected rather than generic: the two
// parser crates are themselves target-gated in `Cargo.toml`, so
// `TreeSitterParser` cannot even be named in a wasm build. Native additionally
// keeps its parser in a `thread_local` because it is `!Send + !Sync`, whereas
// the re2c parser is constructed per call.

#[cfg(not(target_family = "wasm"))]
mod backend {
    use super::*;
    use std::cell::OnceCell;
    use talkbank_parser::TreeSitterParser;

    thread_local! {
        // `TreeSitterParser` owns a reusable tree-sitter buffer and is
        // `!Send + !Sync`; keep one per thread (rayon parallelism is per-file).
        static PARSER: OnceCell<TreeSitterParser> = const { OnceCell::new() };
    }

    fn with_parser<R>(f: impl FnOnce(&TreeSitterParser) -> R) -> R {
        PARSER.with(|cell| {
            let parser = cell.get_or_init(|| {
                TreeSitterParser::new().expect("tree-sitter CHAT grammar failed to load")
            });
            f(parser)
        })
    }

    pub(super) fn parse(input: &str, errors: &ErrorCollector) -> TbChatFile {
        with_parser(|p| p.parse_chat_file_streaming(input, errors))
    }

    pub(super) fn parse_mor(content: &str, errors: &ErrorCollector) -> Option<MorTier> {
        with_parser(|p| p.parse_mor_tier_fragment(content, 0, errors).into_option())
    }

    pub(super) fn parse_gra(content: &str, errors: &ErrorCollector) -> Option<GraTier> {
        with_parser(|p| p.parse_gra_tier_fragment(content, 0, errors).into_option())
    }
}

#[cfg(target_family = "wasm")]
mod backend {
    use super::*;
    use talkbank_model::ChatParser;
    use talkbank_parser_re2c::Re2cParser;

    pub(super) fn parse(input: &str, errors: &ErrorCollector) -> TbChatFile {
        Re2cParser::new()
            .parse_chat_file(input, 0, errors)
            .into_option()
            .unwrap_or_else(|| TbChatFile::new(Vec::new()))
    }

    pub(super) fn parse_mor(content: &str, errors: &ErrorCollector) -> Option<MorTier> {
        Re2cParser::new()
            .parse_mor_tier(content, 0, errors)
            .into_option()
    }

    pub(super) fn parse_gra(content: &str, errors: &ErrorCollector) -> Option<GraTier> {
        Re2cParser::new()
            .parse_gra_tier(content, 0, errors)
            .into_option()
    }
}

// ---------------------------------------------------------------------------
// Public adapter surface
// ---------------------------------------------------------------------------

/// A parse/validation diagnostic surfaced by `chatter`.
///
/// Exposed to Python as `Diagnostic` so a file loaded with `strict=False` can
/// still be inspected: choosing to load a transcript leniently should not mean
/// giving up the ability to find out what is wrong with it.
#[cfg_attr(
    feature = "pyo3",
    pyclass(name = "Diagnostic", frozen, skip_from_py_object)
)]
#[derive(Clone, Debug)]
pub struct ChatDiagnostic {
    /// chatter's canonical error code (e.g. `"E301"`). This is the identifier
    /// chatter's own error specs and documentation are keyed by, and it is
    /// stable across the variant renames the enum itself has seen.
    pub code: String,
    /// The `ErrorCode` variant name (e.g. `"MissingUTF8Header"`).
    pub name: String,
    /// Whether the diagnostic is an error (vs a warning).
    pub is_error: bool,
    /// Whether this is the `%mor`/word count mismatch that rustling reports
    /// through its own misalignment channel (and so must not raise twice).
    pub is_mor_word_mismatch: bool,
    /// Human-readable message from chatter.
    pub message: String,
    /// Source file path (filled by the caller; empty from the adapter).
    pub file_path: String,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ChatDiagnostic {
    /// chatter's canonical error code, e.g. `"E301"`.
    #[getter]
    fn code(&self) -> String {
        self.code.clone()
    }

    /// The rule's name, e.g. `"MissingUTF8Header"`.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// True for an error, false for a warning.
    #[getter]
    fn is_error(&self) -> bool {
        self.is_error
    }

    /// chatter's human-readable description of the problem.
    #[getter]
    fn message(&self) -> String {
        self.message.clone()
    }

    /// The file the diagnostic came from.
    #[getter]
    fn file_path(&self) -> String {
        self.file_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Diagnostic(code='{}', name='{}', is_error={}, file_path='{}')",
            self.code,
            self.name,
            if self.is_error { "True" } else { "False" },
            self.file_path
        )
    }
}

/// The result of adapting one chatter `ChatFile` into rustling structures.
pub(crate) struct Adapted {
    pub(crate) headers: Headers,
    pub(crate) events: Vec<Utterance>,
    pub(crate) misalignments: Vec<MisalignmentInfo>,
    pub(crate) diagnostics: Vec<ChatDiagnostic>,
}

/// Parse `file_text` with chatter and adapt it into rustling structures.
///
/// * `mor_key`/`gra_key` are the `%`-prefixed tier names to treat as the
///   morphology / grammatical-relation tiers (e.g. `Some("%mor")`), or `None`
///   to disable mor+gra handling.
/// * `validate` requests chatter's semantic validation and alignment checks on
///   top of the parse diagnostics, which are collected either way. Diagnostics
///   are always returned rather than being thrown away outside strict mode:
///   deciding to load a file leniently is not a reason to be unable to find out
///   what is wrong with it.
/// * `may_be_fragment` marks input from the string APIs, where a bare utterance
///   body is accepted and needs a synthetic header envelope to parse. Input
///   loaded from files is never wrapped, so a genuinely missing
///   `@UTF8`/`@Begin` header is reported as itself.
/// * `source_path` is the path the text was read from, when there is one. It
///   names the transcript for the validation rules that are about a
///   transcript's own name -- E531, which requires `@Media`'s filename to match
///   it. `None` means the input genuinely has no file name, which is the case
///   for the string APIs.
///
/// Misalignment detection (word count vs non-clitic mor count) is always
/// performed and returned via `misalignments`, independent of `validate`.
pub(crate) fn parse_with_chatter(
    file_text: &str,
    mor_key: Option<&str>,
    gra_key: Option<&str>,
    validate: bool,
    may_be_fragment: bool,
    source_path: Option<&str>,
) -> Adapted {
    let wrapped =
        (may_be_fragment && !looks_like_transcript(file_text)).then(|| wrap_fragment(file_text));
    let input: &str = wrapped.as_deref().unwrap_or(file_text);

    let parse_errors = ErrorCollector::new();
    let chat_file = backend::parse(input, &parse_errors);

    let mut diagnostics = Vec::new();
    // Parse diagnostics are already in hand: the collector is filled by the
    // parse above whether or not anyone asked to validate, so keeping them
    // costs nothing. Semantic validation is a second pass over the file and
    // stays opt-in.
    collect_diagnostics(&parse_errors.into_vec(), &mut diagnostics);
    if validate {
        let verrors = ErrorCollector::new();
        // `Anonymous` is a deliberate answer rather than a missing one: it
        // turns off the rules about the transcript's own name. File input has
        // a name and says so; string input genuinely has none.
        let name = source_path.map_or(TranscriptName::Anonymous, |p| {
            TranscriptName::for_path(std::path::Path::new(p))
        });
        chat_file.validate(&verrors, name);
        collect_diagnostics(&verrors.into_vec(), &mut diagnostics);
        collect_diagnostics(&chat_file.validate_alignments(), &mut diagnostics);
    }

    let (headers, events, misalignments) = map_chat_file(&chat_file, input, mor_key, gra_key);

    Adapted {
        headers,
        events,
        misalignments,
        diagnostics,
    }
}

/// Wrap a bare CHAT fragment in a minimal `@UTF8`/`@Begin`/`@End` envelope.
///
/// chatter's parser only recognises utterances inside a proper transcript body,
/// so fragments (bare utterance lines, as accepted by `CHAT.from_strs`) must be
/// wrapped. Whole transcripts are never wrapped: sniffing their content instead
/// would hide a genuinely missing `@UTF8`/`@Begin` header behind a synthetic one
/// and report the duplicated envelope rather than the real defect. The wrapped
/// text is only fed to the parser; rustling keeps the original `raw_lines`.
fn wrap_fragment(file_text: &str) -> String {
    format!(
        "@UTF8\n@Begin\n{}\n@End\n",
        file_text.trim_end_matches('\n')
    )
}

/// Whether the text already carries a transcript envelope.
///
/// Such input must not be wrapped even when fragments are allowed: doing so
/// would nest one envelope inside another and turn every structural complaint
/// into a duplicate-`@Begin` report about lines the caller never wrote.
fn looks_like_transcript(file_text: &str) -> bool {
    let start = file_text.trim_start();
    start.starts_with("@UTF8") || start.starts_with("@Begin")
}

fn collect_diagnostics(errors: &[ParseError], out: &mut Vec<ChatDiagnostic>) {
    for e in errors {
        out.push(ChatDiagnostic {
            code: e.code.as_str().to_string(),
            name: format!("{:?}", e.code),
            is_error: matches!(e.severity, Severity::Error),
            // Match the two `%mor`-vs-word codes exactly. chatter has a dozen
            // other `*CountMismatch*` codes (`%gra`, `%pho`, `%sin`, `%mod`,
            // ...) that rustling has no separate channel for, so those must
            // still surface as errors under `strict=True`.
            is_mor_word_mismatch: matches!(
                e.code,
                ErrorCode::MorCountMismatchTooFew | ErrorCode::MorCountMismatchTooMany
            ),
            message: e.message.clone(),
            file_path: String::new(),
        });
    }
}

// ---------------------------------------------------------------------------
// ChatFile -> (Headers, events, misalignments)
// ---------------------------------------------------------------------------

fn map_chat_file(
    chat_file: &TbChatFile,
    input: &str,
    mor_key: Option<&str>,
    gra_key: Option<&str>,
) -> (Headers, Vec<Utterance>, Vec<MisalignmentInfo>) {
    let mut headers = Headers {
        languages: chat_file.languages.iter().map(|c| c.to_string()).collect(),
        participants: build_participants(chat_file),
        options: options_string(chat_file),
        media_data: chat_file.media.as_deref().map(map_media),
        ..Headers::default()
    };

    let mut events = Vec::new();
    let mut misalignments = Vec::new();
    let mut seen_utterance = false;

    // Only the span-less backends need rustling's own line blocks, so pay for
    // them only there. One utterance carrying a real span means the parser
    // records them for the whole file.
    let has_spans = chat_file
        .lines
        .iter()
        .any(|l| matches!(l, Line::Utterance(u) if !u.main.span.is_dummy()));
    let blocks = if has_spans {
        Vec::new()
    } else {
        source_blocks(input)
    };
    let mut utterance_index = 0;

    for line in chat_file.lines.iter() {
        match line {
            Line::Utterance(u) => {
                seen_utterance = true;
                let (utt, mis) =
                    map_utterance(u, input, &blocks, utterance_index, mor_key, gra_key);
                utterance_index += 1;
                if let Some(m) = mis {
                    misalignments.push(m);
                }
                events.push(utt);
            }
            Line::Header { header, .. } => {
                if seen_utterance {
                    if let Some(ch) = map_changeable(header) {
                        events.push(changeable_event(ch));
                    }
                } else {
                    fold_file_header(&mut headers, header);
                }
            }
        }
    }

    (headers, events, misalignments)
}

fn changeable_event(ch: ChangeableHeader) -> Utterance {
    Utterance {
        participant: None,
        tokens: None,
        time_marks: None,
        tiers: None,
        changeable_header: Some(ch),
        mor_tier_name: None,
        gra_tier_name: None,
    }
}

// ---------------------------------------------------------------------------
// Utterance mapping
// ---------------------------------------------------------------------------

fn map_utterance(
    u: &talkbank_model::Utterance,
    input: &str,
    blocks: &[SourceBlock],
    index: usize,
    mor_key: Option<&str>,
    gra_key: Option<&str>,
) -> (Utterance, Option<MisalignmentInfo>) {
    let participant = u.main.speaker.as_str().to_string();

    // Verbatim tier text (keyed by participant + %tier).
    let mut tiers: HashMap<String, String> = HashMap::new();
    let (main_text, dep_tiers) = verbatim_text(u, input, blocks, index);
    tiers.insert(participant.clone(), main_text.clone());
    tiers.extend(dep_tiers);

    // chatter models only the utterance-final media bullet; transcripts also
    // carry mid-utterance bullets, so fall back to scanning the raw tier text
    // for the first `\x15start_end\x15` marker (the legacy behavior).
    let time_marks = u
        .main
        .content
        .bullet
        .as_ref()
        .map(|b| (b.timing.start_ms as i64, b.timing.end_ms as i64))
        .or_else(|| first_time_mark(&main_text));

    // Words from the main tier + a trailing terminator token (matches legacy).
    let mut words: Vec<String> = Vec::new();
    collect_words(&u.main.content.content, &mut words);
    if let Some(term) = &u.main.content.terminator {
        words.push(term.to_string());
    }

    // Morphology / grammatical-relation items (from the selected tiers).
    let frag_errors = ErrorCollector::new();
    let mor_items = find_mor_items(u, mor_key, &frag_errors);
    let gra_items = find_gra_items(u, gra_key, &frag_errors);

    let (tokens, counts) = build_tokens(words, mor_items.as_deref(), gra_items.as_deref());

    let misalignment = counts.map(|c: MisalignmentCounts| MisalignmentInfo {
        file_path: String::new(),
        participant: participant.clone(),
        main_tier: main_text.clone(),
        mor_tier_name: mor_key.unwrap_or("%mor").to_string(),
        mor_tier_content: mor_key
            .and_then(|k| tiers.get(k))
            .cloned()
            .unwrap_or_default(),
        word_count: c.word_count,
        mor_count: c.mor_count,
        words: c.words,
        mor_labels: c.mor_labels,
    });

    (
        Utterance {
            participant: Some(participant),
            tokens: Some(tokens),
            time_marks,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: mor_key.map(str::to_string),
            gra_tier_name: gra_key.map(str::to_string),
        },
        misalignment,
    )
}

/// Find the dependent tier named `key` (e.g. `"%mor"`).
///
/// `dependent_tiers` holds `DependentTierEntry` (tier + separator provenance).
/// Since chatter v0.9.0 the entry forwards `kind()`, so the wrapper is matched
/// on directly rather than being unwrapped to its `.tier` first. `kind()` is
/// the bare name (`mor`), so the `%` prefix is stripped from the key rather
/// than formatted onto every tier name.
fn find_tier<'a>(u: &'a talkbank_model::Utterance, key: Option<&str>) -> Option<&'a DependentTier> {
    let name = key?.strip_prefix('%')?;
    u.dependent_tiers
        .iter()
        .find(|entry| entry.kind() == name)
        .map(|entry| &entry.tier)
}

/// Morphology items for the tier selected by `mor_key`, parsing custom `%x…`
/// tiers that chatter does not type as `%mor`.
fn find_mor_items(
    u: &talkbank_model::Utterance,
    mor_key: Option<&str>,
    errors: &ErrorCollector,
) -> Option<Vec<MorItem>> {
    // The terminator item pairs with the terminator word `map_utterance`
    // appends to `words`, and that word is only appended when the main tier
    // actually carries one. chatter's `MorTier::terminator` is not optional, so
    // an utterance written without a terminator would otherwise get a mor item
    // with no word to match and come out one too long on every count.
    let with_terminator = u.main.content.terminator.is_some();
    match find_tier(u, mor_key)? {
        DependentTier::Mor(m) => Some(mor_items_from_tier(m, with_terminator)),
        DependentTier::UserDefined(t) | DependentTier::Unsupported(t) => {
            // `content` is `Option` as of chatter v0.9.0: a `%x` line that
            // declared a tier and gave it nothing has no items to parse.
            backend::parse_mor(t.content.as_ref()?.as_str(), errors)
                .map(|tier| mor_items_from_tier(&tier, with_terminator))
        }
        _ => None,
    }
}

/// Grammatical-relation items for the tier selected by `gra_key`.
fn find_gra_items(
    u: &talkbank_model::Utterance,
    gra_key: Option<&str>,
    errors: &ErrorCollector,
) -> Option<Vec<Gra>> {
    match find_tier(u, gra_key)? {
        DependentTier::Gra(g) => Some(gra_items_from_tier(g)),
        DependentTier::UserDefined(t) | DependentTier::Unsupported(t) => {
            backend::parse_gra(t.content.as_ref()?.as_str(), errors)
                .map(|tier| gra_items_from_tier(&tier))
        }
        _ => None,
    }
}

/// Flatten a chatter `MorTier` into rustling's legacy `MorItem` sequence:
/// each `Mor` contributes its main word (non-clitic) then its post-clitics
/// (clitic), and a final non-clitic item carries the tier terminator so the
/// count matches the trailing terminator word in `words`.
///
/// `with_terminator` is false when the utterance's main tier has no terminator
/// and so contributes no terminator word for that item to pair with.
fn mor_items_from_tier(tier: &MorTier, with_terminator: bool) -> Vec<MorItem> {
    let mut items = Vec::new();
    for mor in tier.items() {
        let (pos, m) = render_mor(&mor.main);
        items.push(MorItem {
            pos,
            mor: m,
            is_clitic: false,
        });
        for clitic in mor.post_clitics.iter() {
            let (pos, m) = render_mor(clitic);
            items.push(MorItem {
                pos,
                mor: m,
                is_clitic: true,
            });
        }
    }
    if with_terminator {
        items.push(MorItem {
            pos: String::new(),
            mor: tier.terminator.to_string(),
            is_clitic: false,
        });
    }
    items
}

fn gra_items_from_tier(tier: &GraTier) -> Vec<Gra> {
    tier.relations()
        .iter()
        .map(|r| Gra {
            dep: r.index,
            head: r.head,
            rel: r.relation.as_str().to_string(),
        })
        .collect()
}

/// Render a `MorWord` as rustling's `(pos, mor)` pair.
///
/// chatter writes a `%mor` item as `POS|lemma[-Feature]*`; rustling's `Token`
/// keeps the two halves in separate fields. Both are read directly off the
/// typed model -- `pos` from the field, the analysis from `analysis()` (added
/// in chatter v0.9.0) -- so neither half is recovered by splitting a rendered
/// string, and the split point cannot drift from the value it describes.
fn render_mor(mw: &MorWord) -> (String, String) {
    (mw.pos.to_string(), mw.analysis().to_string())
}

// ---------------------------------------------------------------------------
// Word collection (token text extraction)
// ---------------------------------------------------------------------------

fn push_word(w: &Word, out: &mut Vec<String>) {
    // `counts_for_tier` is chatter's canonical %mor-alignability gate (drops
    // omissions, fillers, fragments, nonwords and untranscribed xxx/yyy/www).
    // Deferring to it keeps our word count identical to the count chatter's own
    // alignment validation uses, instead of duplicating the category list here.
    if !counts_for_tier(w, TierDomain::Mor) {
        return;
    }
    let cleaned = w.cleaned_text();
    if !cleaned.is_empty() {
        out.push(cleaned.to_string());
    }
}

/// Push the words a `[: replacement]` contributes to `%mor` alignment.
///
/// `%mor` follows the corrected transcript slot, so the replacement words align
/// when present and the original surface word only when there is no
/// replacement. `[e]`-excluded material contributes nothing.
fn push_replaced_word(r: &ReplacedWord, out: &mut Vec<String>) {
    if annotations_have_alignment_ignore(&r.scoped_annotations) {
        return;
    }
    if r.replacement.words.is_empty() {
        push_word(&r.word, out);
    } else {
        for w in r.replacement.words.iter() {
            push_word(w, out);
        }
    }
}

fn push_separator(s: &Separator, out: &mut Vec<String>) {
    // Only tag-marker separators carry `%mor` items (Tag -> end|end, Vocative ->
    // beg|beg, Comma -> cm|cm); the rest are punctuation with no mor item and
    // must not become word tokens.
    if !is_tag_marker_separator(s) {
        return;
    }
    let mut buf = String::new();
    let _ = s.write_chat(&mut buf);
    let text = buf.trim();
    if !text.is_empty() {
        out.push(text.to_string());
    }
}

/// Collect the main-tier words that align with `%mor`.
///
/// Traversal is delegated to chatter's `walk_words` with [`TierDomain::Mor`], so
/// the domain rules (descend into groups/quotations/phonological groups, skip
/// retraces and `[e]`-excluded material) stay in sync with the alignment
/// policy chatter validates against, rather than being re-derived here.
fn collect_words(items: &[UtteranceContent], out: &mut Vec<String>) {
    walk_words(items, Some(TierDomain::Mor), &mut |item| match item {
        WordItem::Word(w) => push_word(w, out),
        WordItem::ReplacedWord(r) => push_replaced_word(r, out),
        WordItem::Separator(s) => push_separator(s, out),
    });
}

// ---------------------------------------------------------------------------
// Verbatim tier text
// ---------------------------------------------------------------------------

/// One utterance's source lines: the main tier and its dependent tiers, with
/// continuation lines folded back in.
///
/// This is rustling's own reading of the source, used only to display text that
/// chatter's spans cannot reach. It never feeds the parser.
#[derive(Default)]
pub(crate) struct SourceBlock {
    main: String,
    tiers: Vec<(String, String)>,
}

/// Where a continuation line belongs: the tier the most recent non-indented
/// line opened, or nothing at all when that line was one this reading does not
/// track (a header or `@End`).
#[derive(Clone, Copy)]
enum Continuation {
    Main,
    Dependent,
    Untracked,
}

/// Split the source into one [`SourceBlock`] per main tier, in file order.
///
/// Only built when the parser records no spans; see [`verbatim_text`].
fn source_blocks(input: &str) -> Vec<SourceBlock> {
    let mut blocks: Vec<SourceBlock> = Vec::new();
    let mut continuation = Continuation::Untracked;
    for line in input.lines() {
        if line.starts_with([' ', '\t']) {
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if let Some(block) = blocks.last_mut() {
                let target = match continuation {
                    Continuation::Main => Some(&mut block.main),
                    Continuation::Dependent => block.tiers.last_mut().map(|(_, content)| content),
                    Continuation::Untracked => None,
                };
                if let Some(target) = target {
                    target.push(' ');
                    target.push_str(text);
                }
            }
            continue;
        }
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix('*') {
            continuation = Continuation::Main;
            blocks.push(SourceBlock {
                main: tier_content(rest),
                tiers: Vec::new(),
            });
        } else if line.starts_with('%') {
            // Only a tier that was actually opened can take a continuation:
            // otherwise the text would land on whichever tier came before.
            continuation = Continuation::Untracked;
            if let Some((name, content)) = tier_name_and_content(line)
                && let Some(block) = blocks.last_mut()
            {
                block.tiers.push((name, content));
                continuation = Continuation::Dependent;
            }
        } else {
            // A header or `@End`. Its own continuation lines belong to it, not
            // to the utterance above, which is where they used to be appended:
            // a wrapped `@Comment` surfaced as speech on the previous *tier.
            continuation = Continuation::Untracked;
        }
    }
    blocks
}

/// Verbatim `(main tier text, dependent tiers)` for one utterance.
///
/// Read by slicing the source with chatter's spans where the parser records
/// them. The re2c backend, which is what wasm/Pyodide builds use, records none
/// at all -- its lexer produces spans and the parser discards them -- so every
/// span is [`Span::is_dummy`] there and slicing yields `""`. Falling back to
/// rustling's own line blocks keeps `annotated`, `audible` and `tiers`
/// populated on both backends instead of only the native one.
fn verbatim_text(
    u: &talkbank_model::Utterance,
    input: &str,
    blocks: &[SourceBlock],
    index: usize,
) -> (String, Vec<(String, String)>) {
    if !u.main.span.is_dummy() {
        let main = tier_content(slice(input, u.main.span));
        let deps = u
            .dependent_tiers
            .iter()
            .filter_map(|dt| tier_name_and_content(slice(input, dt.span())))
            .collect();
        return (main, deps);
    }
    match blocks.get(index) {
        Some(block) => (block.main.clone(), block.tiers.clone()),
        None => (String::new(), Vec::new()),
    }
}

fn slice(input: &str, span: talkbank_model::Span) -> &str {
    input.get(span.to_range()).unwrap_or("")
}

/// Extract the first `\x15start_end\x15` media bullet from raw tier text.
///
/// chatter's typed model exposes only the utterance-final bullet; transcripts
/// aligned phrase by phrase carry bullets mid-utterance, which this recovers.
fn first_time_mark(text: &str) -> Option<(i64, i64)> {
    const BULLET: char = '\u{15}';
    let after_open = &text[text.find(BULLET)? + BULLET.len_utf8()..];
    let inner = &after_open[..after_open.find(BULLET)?];
    let (start, end) = inner.split_once('_')?;
    Some((start.trim().parse().ok()?, end.trim().parse().ok()?))
}

/// Trim a raw tier slice and fold its continuation lines into single spaces.
///
/// CHAT wraps a long tier by starting the next line with a tab, so a slice cut
/// from the source with chatter's span carries that newline and indent
/// verbatim. `annotated`, `audible` and the `tiers` map are all single-line
/// views of a tier, and the writers reading them treat a newline as a record
/// boundary -- an embedded one splits an SRT subtitle in two and terminates a
/// TextGrid quoted string early. [`source_blocks`], the fallback used where the
/// parser records no spans, folds the same way, so both backends agree.
fn fold_continuations(raw: &str) -> String {
    if !raw.contains('\n') {
        return raw.trim().to_string();
    }
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract the content after `*CODE:` / `%tier:` from a raw tier line slice.
fn tier_content(raw: &str) -> String {
    raw.split_once(':')
        .map(|(_, rest)| fold_continuations(rest))
        .unwrap_or_default()
}

/// Split a raw dependent-tier line slice into (`%name`, content).
fn tier_name_and_content(raw: &str) -> Option<(String, String)> {
    let (name, content) = raw.split_once(':')?;
    Some((name.trim().to_string(), fold_continuations(content)))
}

// ---------------------------------------------------------------------------
// Header mapping
// ---------------------------------------------------------------------------

fn opt_nonempty(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn map_age(age: &AgeValue) -> Option<Age> {
    match age {
        AgeValue::Valid {
            years,
            months,
            days,
            ..
        } => Some(Age {
            years: *years as u32,
            months: months.map(|m| m as u32),
            days: days.map(|d| d as u32),
        }),
        AgeValue::Unsupported(_) => None,
    }
}

/// Build the participant list from the `@Participants` header entries, enriched
/// with `@ID` metadata from chatter's merged participant map.
///
/// chatter only populates `chat_file.participants` when `@ID` headers are
/// present, so `@Participants`-only files would otherwise lose their speaker
/// list. The `@Participants` entries are the authoritative declared order.
fn build_participants(chat_file: &TbChatFile) -> Vec<Participant> {
    let entries = chat_file
        .lines
        .iter()
        .find_map(|line| match line.as_header() {
            Some(Header::Participants { entries }) => Some(entries),
            _ => None,
        });
    let Some(entries) = entries else {
        return chat_file
            .participants
            .values()
            .map(map_participant)
            .collect();
    };
    entries
        .iter()
        .map(
            |entry| match chat_file.participants.get(&entry.speaker_code) {
                Some(p) => map_participant(p),
                None => Participant {
                    code: entry.speaker_code.as_str().to_string(),
                    name: entry
                        .name
                        .as_ref()
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                    role: entry.role.as_str().to_string(),
                    ..Participant::default()
                },
            },
        )
        .collect()
}

fn map_participant(p: &talkbank_model::Participant) -> Participant {
    let id = &p.id;
    Participant {
        code: p.code.as_str().to_string(),
        name: p.name.as_ref().map(|n| n.to_string()).unwrap_or_default(),
        role: p.role.as_str().to_string(),
        // `@ID` carries a comma-separated language list for bilingual speakers;
        // keep every code rather than only the first.
        language: opt_nonempty(
            &id.language
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
        corpus: opt_nonempty(id.corpus.as_str()),
        age: id.age.as_ref().and_then(map_age),
        sex: id.sex.as_ref().map(|s| s.as_str().to_string()),
        group: id.group.as_ref().and_then(|g| opt_nonempty(g.as_str())),
        ses: id.ses.as_ref().and_then(|s| opt_nonempty(&s.to_string())),
        education: id.education.as_ref().and_then(|e| opt_nonempty(e.as_str())),
        custom: id
            .custom_field
            .as_ref()
            .and_then(|c| opt_nonempty(c.as_str())),
        birth: p.birth_date.as_ref().map(|d| d.to_string()),
        birthplace: None,
        l1: None,
    }
}

fn map_media(m: &talkbank_model::MediaHeader) -> Media {
    Media {
        filename: m.filename.to_string(),
        format: m.media_type.as_str().to_string(),
        status: m.status.as_ref().map(|s| s.as_str().to_string()),
    }
}

fn options_string(chat_file: &TbChatFile) -> Option<String> {
    let opts: Vec<String> = chat_file
        .options
        .iter()
        .map(|o| o.as_str().to_string())
        .collect();
    if opts.is_empty() {
        None
    } else {
        Some(opts.join(", "))
    }
}

/// Fold a pre-first-utterance header into the file-level [`Headers`].
///
/// `@UTF8`/`@Begin`/`@End`/`@Languages`/`@Participants`/`@ID`/`@Options`/`@Media`
/// are handled elsewhere (or synthetic envelope) and skipped here.
fn fold_file_header(headers: &mut Headers, header: &Header) {
    match header {
        Header::Pid { pid } => headers.pid = Some(pid.to_string()),
        Header::Date { date } => headers.date = Some(date.to_string()),
        Header::Comment { content } => headers
            .comments
            .get_or_insert_with(Vec::new)
            .push(bullet_text(content)),
        Header::Situation { text } => headers.situation = Some(text.to_string()),
        Header::Types(t) => {
            headers.types = Some(format!(
                "{}, {}, {}",
                t.design.as_str(),
                t.activity.as_str(),
                t.group.as_str()
            ))
        }
        Header::Number { number } => headers.number = Some(number.as_str().to_string()),
        Header::RecordingQuality { quality } => {
            headers.recording_quality = Some(quality.as_str().to_string())
        }
        Header::Transcription { transcription } => {
            headers.transcription = Some(transcription.as_str().to_string())
        }
        Header::TapeLocation { location } => headers.tape_location = Some(location.to_string()),
        Header::TimeDuration { duration } => headers.time_duration = Some(duration.to_string()),
        Header::TimeStart { start } => headers.time_start = Some(start.to_string()),
        Header::Location { location } => headers.location = Some(location.to_string()),
        Header::RoomLayout { layout } => headers.room_layout = Some(layout.to_string()),
        Header::Transcriber { transcriber } => headers.transcriber = Some(transcriber.to_string()),
        Header::Warning { text } => headers.warning = Some(text.to_string()),
        Header::Videos { videos } => headers.videos = Some(videos.to_string()),
        Header::Birth { participant, date } => {
            set_participant_field(headers, participant.as_str(), |p| {
                p.birth = Some(date.to_string())
            })
        }
        Header::Birthplace { participant, place } => {
            set_participant_field(headers, participant.as_str(), |p| {
                p.birthplace = Some(place.to_string())
            })
        }
        Header::L1Of {
            participant,
            language,
        } => set_participant_field(headers, participant.as_str(), |p| {
            p.l1 = Some(language.to_string())
        }),
        Header::Unknown { text, .. } => {
            if let Some((name, value)) = text.as_str().split_once(':') {
                headers
                    .other
                    .insert(name.trim().to_string(), value.trim().to_string());
            }
        }
        // Structural / already-handled / non-file headers.
        _ => {}
    }
}

fn set_participant_field(headers: &mut Headers, code: &str, f: impl FnOnce(&mut Participant)) {
    if let Some(p) = headers.participants.iter_mut().find(|p| p.code == code) {
        f(p);
    }
}

/// Map a mid-file header to a rustling changeable header, or `None` for headers
/// that are not changeable (and so dropped from the event stream).
fn map_changeable(header: &Header) -> Option<ChangeableHeader> {
    Some(match header {
        Header::Comment { content } => ChangeableHeader::Comment {
            value: bullet_text(content),
        },
        Header::Situation { text } => ChangeableHeader::Situation {
            value: text.to_string(),
        },
        Header::Date { date } => ChangeableHeader::Date {
            value: date.to_string(),
        },
        Header::NewEpisode => ChangeableHeader::NewEpisode {},
        Header::Blank => ChangeableHeader::Blank {},
        Header::BeginGem { label } => ChangeableHeader::Bg {
            value: label.as_ref().map(|l| l.to_string()),
        },
        Header::EndGem { label } => ChangeableHeader::Eg {
            value: label.as_ref().map(|l| l.to_string()),
        },
        Header::LazyGem { label } => ChangeableHeader::G {
            value: label.as_ref().map(|l| l.to_string()),
        },
        Header::Activities { activities } => ChangeableHeader::Activities {
            value: activities.to_string(),
        },
        Header::Bck { bck } => ChangeableHeader::Bck {
            value: bck.to_string(),
        },
        Header::Page { page } => ChangeableHeader::Page {
            value: page.to_string(),
        },
        _ => return None,
    })
}

/// Extract the plain text of a `@Comment`'s bullet content (ignoring bullets).
fn bullet_text(content: &talkbank_model::BulletContent) -> String {
    use talkbank_model::BulletContentSegment;
    content
        .segments
        .iter()
        .filter_map(|seg| match seg {
            BulletContentSegment::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    fn parse(text: &str) -> Adapted {
        parse_with_chatter(text, Some("%mor"), Some("%gra"), false, true, None)
    }

    const BASIC: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
        @Participants:\tCHI Target_Child\n\
        @ID:\teng|test|CHI|2;10.05|female|||Target_Child|||\n\
        *CHI:\tI want cookie .\n%mor:\tpro|I v|want n|cookie .\n\
        %gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n@End\n";

    #[test]
    fn basic_tokens_tiers_headers() {
        let a = parse(BASIC);
        assert_eq!(a.events.len(), 1);
        let u = &a.events[0];
        assert_eq!(u.participant.as_deref(), Some("CHI"));
        let tokens = u.tokens.as_ref().unwrap();
        let words: Vec<&str> = tokens.iter().map(|t| t.word.as_str()).collect();
        assert_eq!(words, ["I", "want", "cookie", "."]);
        assert_eq!(tokens[0].pos.as_deref(), Some("pro"));
        assert_eq!(tokens[0].mor.as_deref(), Some("I"));
        assert_eq!(tokens[0].gra.as_ref().unwrap().rel, "SUBJ");
        assert_eq!(tokens[3].pos.as_deref(), Some("")); // terminator token
        assert_eq!(tokens[3].mor.as_deref(), Some("."));
        assert_eq!(tokens[3].gra.as_ref().unwrap().rel, "PUNCT");
        assert_eq!(u.annotated().as_deref(), Some("I want cookie ."));
        let tiers = u.tiers.as_ref().unwrap();
        assert_eq!(
            tiers.get("CHI").map(String::as_str),
            Some("I want cookie .")
        );
        assert_eq!(
            tiers.get("%mor").map(String::as_str),
            Some("pro|I v|want n|cookie .")
        );
        assert_eq!(a.headers.languages, ["eng"]);
        let p = &a.headers.participants[0];
        assert_eq!(p.code, "CHI");
        assert_eq!(p.role, "Target_Child");
        assert_eq!(p.sex.as_deref(), Some("female"));
        assert_eq!(
            p.age,
            Some(Age {
                years: 2,
                months: Some(10),
                days: Some(5)
            })
        );
        assert!(a.misalignments.is_empty());
    }

    #[test]
    fn postclitic_empty_word_token() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\tthat's mine .\n%mor:\tpro:dem|that~cop|be&3S pro:poss:det|mine .\n@End\n";
        let a = parse(text);
        let tokens = a.events[0].tokens.as_ref().unwrap();
        // that's -> that / clitic be&3S / mine / .
        assert_eq!(tokens[0].word, "that's");
        assert_eq!(tokens[0].pos.as_deref(), Some("pro:dem"));
        assert_eq!(tokens[1].word, ""); // clitic has empty word
        assert_eq!(tokens[1].pos.as_deref(), Some("cop"));
        assert_eq!(tokens[1].mor.as_deref(), Some("be&3S"));
        assert_eq!(tokens[2].word, "mine");
        assert!(a.misalignments.is_empty());
    }

    #[test]
    fn time_marks_parsed() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\thello . \u{15}123_456\u{15}\n@End\n";
        let a = parse(text);
        assert_eq!(a.events[0].time_marks, Some((123, 456)));
    }

    #[test]
    fn custom_xmor_xgra_tiers() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\tI want cookie .\n%xmor:\tpro|I v|want n|cookie .\n\
            %xgra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n@End\n";
        let a = parse_with_chatter(text, Some("%xmor"), Some("%xgra"), false, true, None);
        let tokens = a.events[0].tokens.as_ref().unwrap();
        assert_eq!(tokens[0].pos.as_deref(), Some("pro"));
        assert_eq!(tokens[0].gra.as_ref().unwrap().rel, "SUBJ");
        assert_eq!(a.events[0].mor_tier_name.as_deref(), Some("%xmor"));
        // Default %mor is absent, so with default keys tokens have no pos.
        let b = parse(text);
        assert!(b.events[0].tokens.as_ref().unwrap()[0].pos.is_none());
    }

    #[test]
    fn misalignment_detected() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\tI want cookie .\n%mor:\tpro|I v|want .\n@End\n";
        let a = parse(text);
        assert_eq!(a.misalignments.len(), 1);
        assert!(a.events[0].tokens.as_ref().unwrap().is_empty());
        // tiers preserved even on misalignment
        assert!(a.events[0].tiers.as_ref().unwrap().contains_key("%mor"));
    }

    #[test]
    fn changeable_headers_ordered() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\thello .\n@New Episode\n@Comment:\tChild laughs\n*CHI:\tbye .\n@End\n";
        let a = parse(text);
        assert_eq!(a.events.len(), 4);
        assert_eq!(a.events[0].participant.as_deref(), Some("CHI"));
        assert!(matches!(
            a.events[1].changeable_header,
            Some(ChangeableHeader::NewEpisode {})
        ));
        match &a.events[2].changeable_header {
            Some(ChangeableHeader::Comment { value }) => assert_eq!(value, "Child laughs"),
            other => panic!("expected Comment, got {other:?}"),
        }
        assert_eq!(a.events[3].participant.as_deref(), Some("CHI"));
    }

    #[test]
    fn bare_fragment_is_wrapped_and_parsed() {
        let a = parse("*CHI:\tI want cookie .");
        assert_eq!(a.events.len(), 1);
        let words: Vec<&str> = a.events[0]
            .tokens
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.word.as_str())
            .collect();
        assert_eq!(words, ["I", "want", "cookie", "."]);
    }

    #[test]
    fn no_mor_tier_gives_wordonly_tokens() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tno .\n@End\n";
        let a = parse(text);
        let tokens = a.events[0].tokens.as_ref().unwrap();
        let words: Vec<&str> = tokens.iter().map(|t| t.word.as_str()).collect();
        assert_eq!(words, ["no", "."]);
        assert!(tokens[0].pos.is_none());
        assert!(tokens[0].mor.is_none());
    }

    /// A main tier without a terminator contributes no terminator word, so the
    /// `%mor` tier must not contribute a terminator item either. chatter's
    /// `MorTier::terminator` is not optional, so the item used to be appended
    /// unconditionally and every such utterance came out one mor item long --
    /// reported as a misalignment chatter itself does not see, and emptying the
    /// tokens under `strict=False`. chatter reports the real fault (E305) on
    /// its own.
    #[test]
    fn missing_main_terminator_does_not_inflate_mor_count() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\tI want cookie\n%mor:\tpro|I v|want n|cookie .\n@End\n";
        let a = parse(text);
        let tokens = a.events[0].tokens.as_ref().unwrap();
        let words: Vec<&str> = tokens.iter().map(|t| t.word.as_str()).collect();
        assert_eq!(words, ["I", "want", "cookie"]);
        assert_eq!(tokens[2].pos.as_deref(), Some("n"));
        assert!(a.misalignments.is_empty());
    }

    /// A tier wrapped over several source lines is a single-line view once
    /// read. The span cut from the source carries the newline and the tab that
    /// continues the tier, and a newline reaching `annotated`/`tiers` splits an
    /// SRT subtitle and terminates a TextGrid quoted string early.
    #[test]
    fn wrapped_tier_lines_are_folded() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
            @ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\tI want\n\ta cookie .\n%com:\tsaid while\n\tpointing .\n@End\n";
        let a = parse(text);
        let u = &a.events[0];
        assert_eq!(u.annotated().as_deref(), Some("I want a cookie ."));
        let tiers = u.tiers.as_ref().unwrap();
        assert_eq!(
            tiers.get("CHI").map(String::as_str),
            Some("I want a cookie .")
        );
        assert_eq!(
            tiers.get("%com").map(String::as_str),
            Some("said while pointing .")
        );
        for text in tiers.values() {
            assert!(!text.contains('\n'), "tier text kept a newline: {text:?}");
        }
    }

    /// The span-less fallback reads the source by line. A continuation line
    /// belongs to the tier above it, and a header's continuation belongs to the
    /// header -- not to the last utterance, where it used to be appended and
    /// surfaced as speech.
    #[test]
    fn source_blocks_ignore_header_continuations() {
        let input = "@UTF8\n@Begin\n*CHI:\tI want\n\ta cookie .\n%com:\tpointing\n\tat it .\n\
            @Comment:\tlong note\n\tcontinued here\n@End\n";
        let blocks = source_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].main, "I want a cookie .");
        assert_eq!(
            blocks[0].tiers,
            vec![("%com".to_string(), "pointing at it .".to_string())]
        );
    }
}
