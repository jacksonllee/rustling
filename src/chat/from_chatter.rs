//! Adapter from TalkBank `chatter`'s typed CHAT model into rustling's CHAT
//! data structures.
//!
//! CHAT parsing is delegated to the official `chatter` crates
//! (`talkbank-model` + a target-specific parser backend); this module maps the
//! resulting typed [`talkbank_model::ChatFile`] into rustling's existing
//! [`Headers`]/[`Utterance`]/[`Token`] structures so the Python API is
//! unchanged.

use std::collections::HashMap;

use crate::chat::header::{Age, ChangeableHeader, Headers, Media, Participant};
use crate::chat::reader::{MisalignmentCounts, MisalignmentInfo, MorItem, build_tokens};
use crate::chat::utterance::{Gra, Utterance};

use talkbank_model::{
    AgeValue, BracketedItem, ChatFile as TbChatFile, DependentTier, ErrorCollector, GraTier,
    Header, Line, MorTier, MorWord, ParseError, Separator, Severity, UtteranceContent, Word,
    WriteChat,
};

// ---------------------------------------------------------------------------
// Parser backend (target-gated)
// ---------------------------------------------------------------------------
//
// `TreeSitterParser` (native) and `Re2cParser` (wasm) expose different APIs and
// only the latter implements the `ChatParser` trait, so the backend is a
// cfg-selected trio of free functions rather than a shared trait object.

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

/// A parse/validation diagnostic surfaced by `chatter`, reduced to the fields
/// rustling's strict-mode policy needs.
#[derive(Clone, Debug)]
pub(crate) struct ChatDiagnostic {
    /// The `ErrorCode` variant name (e.g. `"MissingUTF8Header"`).
    pub(crate) code: String,
    /// Whether the diagnostic is an error (vs a warning).
    pub(crate) is_error: bool,
    /// Human-readable message from chatter.
    pub(crate) message: String,
    /// Source file path (filled by the caller; empty from the adapter).
    pub(crate) file_path: String,
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
/// * When `validate` is true, chatter's semantic validation + alignment checks
///   run and their diagnostics are returned (used only for `strict=True`).
///
/// Misalignment detection (word count vs non-clitic mor count) is always
/// performed and returned via `misalignments`, independent of `validate`.
pub(crate) fn parse_with_chatter(
    file_text: &str,
    mor_key: Option<&str>,
    gra_key: Option<&str>,
    validate: bool,
) -> Adapted {
    let wrapped = maybe_wrap(file_text);
    let input: &str = wrapped.as_deref().unwrap_or(file_text);

    let parse_errors = ErrorCollector::new();
    let chat_file = backend::parse(input, &parse_errors);

    let mut diagnostics = Vec::new();
    if validate {
        collect_diagnostics(&parse_errors.into_vec(), &mut diagnostics);
        let verrors = ErrorCollector::new();
        chat_file.validate(&verrors, None);
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
/// chatter's tree-sitter parser only recognises utterances inside a proper
/// transcript body, so fragments (common in `CHAT.from_strs`) must be wrapped.
/// Full files (starting with `@UTF8`) are parsed as-is. The wrapped text is
/// only fed to the parser; rustling keeps the original `raw_lines`.
fn maybe_wrap(file_text: &str) -> Option<String> {
    if file_text.trim_start().starts_with("@UTF8") {
        None
    } else {
        Some(format!(
            "@UTF8\n@Begin\n{}\n@End\n",
            file_text.trim_end_matches('\n')
        ))
    }
}

fn collect_diagnostics(errors: &[ParseError], out: &mut Vec<ChatDiagnostic>) {
    for e in errors {
        out.push(ChatDiagnostic {
            code: format!("{:?}", e.code),
            is_error: matches!(e.severity, Severity::Error),
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

    for line in chat_file.lines.iter() {
        match line {
            Line::Utterance(u) => {
                seen_utterance = true;
                let (utt, mis) = map_utterance(u, input, mor_key, gra_key);
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
    mor_key: Option<&str>,
    gra_key: Option<&str>,
) -> (Utterance, Option<MisalignmentInfo>) {
    let participant = u.main.speaker.as_str().to_string();

    // Verbatim tier text (keyed by participant + %tier), sliced from source.
    let mut tiers: HashMap<String, String> = HashMap::new();
    let main_text = tier_content(slice(input, u.main.span));
    tiers.insert(participant.clone(), main_text.clone());
    for dt in u.dependent_tiers.iter() {
        if let Some((name, content)) = tier_name_and_content(slice(input, dt.span())) {
            tiers.insert(name, content);
        }
    }

    let time_marks = u
        .main
        .content
        .bullet
        .as_ref()
        .map(|b| (b.timing.start_ms as i64, b.timing.end_ms as i64));

    // Words from the main tier + a trailing terminator token (matches legacy).
    let mut words: Vec<String> = Vec::new();
    collect_words_uc(&u.main.content.content, &mut words);
    if let Some(term) = &u.main.content.terminator {
        words.push(term.to_string());
    }

    // Morphology / grammatical-relation items (from the selected tiers).
    let frag_errors = ErrorCollector::new();
    let mor_items = find_mor_tier(u, mor_key, &frag_errors).map(|tier| mor_items_from_tier(&tier));
    let gra_items = find_gra_tier(u, gra_key, &frag_errors).map(|tier| gra_items_from_tier(&tier));

    let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let (tokens, counts) = build_tokens(&word_refs, mor_items.as_deref(), gra_items.as_deref());

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

/// Find the morphology tier selected by `mor_key`, parsing custom `%x…` tiers.
fn find_mor_tier(
    u: &talkbank_model::Utterance,
    mor_key: Option<&str>,
    errors: &ErrorCollector,
) -> Option<MorTier> {
    let key = mor_key?;
    for dt in u.dependent_tiers.iter() {
        if format!("%{}", dt.kind()) != key {
            continue;
        }
        return match dt {
            DependentTier::Mor(m) => Some(m.clone()),
            DependentTier::UserDefined(t) | DependentTier::Unsupported(t) => {
                backend::parse_mor(t.content.as_str(), errors)
            }
            _ => None,
        };
    }
    None
}

fn find_gra_tier(
    u: &talkbank_model::Utterance,
    gra_key: Option<&str>,
    errors: &ErrorCollector,
) -> Option<GraTier> {
    let key = gra_key?;
    for dt in u.dependent_tiers.iter() {
        if format!("%{}", dt.kind()) != key {
            continue;
        }
        return match dt {
            DependentTier::Gra(g) => Some(g.clone()),
            DependentTier::UserDefined(t) | DependentTier::Unsupported(t) => {
                backend::parse_gra(t.content.as_str(), errors)
            }
            _ => None,
        };
    }
    None
}

/// Flatten a chatter `MorTier` into rustling's legacy `MorItem` sequence:
/// each `Mor` contributes its main word (non-clitic) then its post-clitics
/// (clitic), and a final non-clitic item carries the tier terminator so the
/// count matches the trailing terminator word in `words`.
fn mor_items_from_tier(tier: &MorTier) -> Vec<MorItem> {
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
    items.push(MorItem {
        pos: String::new(),
        mor: tier.terminator.to_string(),
        is_clitic: false,
    });
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

/// Render a `MorWord` as legacy `(pos, mor)`: chatter writes `pos|lemma-feat…`,
/// split at the first `|`.
fn render_mor(mw: &MorWord) -> (String, String) {
    let mut s = String::new();
    let _ = mw.write_chat(&mut s);
    match s.split_once('|') {
        Some((pos, mor)) => (pos.to_string(), mor.to_string()),
        None => (String::new(), s),
    }
}

// ---------------------------------------------------------------------------
// Word collection (token text extraction)
// ---------------------------------------------------------------------------

fn push_word(w: &Word, out: &mut Vec<String>) {
    // Drop unintelligible (xxx/yyy/www) and non-spoken word categories, matching
    // the legacy word filter; keep everything else via chatter's cleaned text.
    if w.untranscribed().is_some() {
        return;
    }
    if let Some(cat) = &w.category {
        use talkbank_model::WordCategory::*;
        if matches!(
            cat,
            Omission | Filler | PhonologicalFragment | Nonword | CAOmission
        ) {
            return;
        }
    }
    let cleaned = w.cleaned_text();
    if !cleaned.is_empty() {
        out.push(cleaned.to_string());
    }
}

fn push_separator(s: &Separator, out: &mut Vec<String>) {
    // CA separators (commas, etc.) align with `cm|cm`-style mor items, so they
    // surface as word tokens.
    let mut buf = String::new();
    let _ = s.write_chat(&mut buf);
    let text = buf.trim();
    if !text.is_empty() {
        out.push(text.to_string());
    }
}

fn collect_words_uc(items: &[UtteranceContent], out: &mut Vec<String>) {
    for item in items {
        match item {
            UtteranceContent::Word(w) => push_word(w, out),
            UtteranceContent::AnnotatedWord(a) => push_word(&a.inner, out),
            UtteranceContent::ReplacedWord(r) => {
                for w in r.replacement.words.0.iter() {
                    push_word(w, out);
                }
            }
            UtteranceContent::Group(g) => collect_words_bi(&g.content.content, out),
            UtteranceContent::AnnotatedGroup(a) => collect_words_bi(&a.inner.content.content, out),
            UtteranceContent::Quotation(q) => collect_words_bi(&q.content.content, out),
            UtteranceContent::Separator(s) => push_separator(s, out),
            // Retraces are excluded from %mor alignment; other content items
            // (events, pauses, overlaps, markers) carry no word tokens.
            _ => {}
        }
    }
}

fn collect_words_bi(items: &[BracketedItem], out: &mut Vec<String>) {
    for item in items {
        match item {
            BracketedItem::Word(w) => push_word(w, out),
            BracketedItem::AnnotatedWord(a) => push_word(&a.inner, out),
            BracketedItem::ReplacedWord(r) => {
                for w in r.replacement.words.0.iter() {
                    push_word(w, out);
                }
            }
            BracketedItem::AnnotatedGroup(a) => collect_words_bi(&a.inner.content.content, out),
            BracketedItem::Quotation(q) => collect_words_bi(&q.content.content, out),
            BracketedItem::Separator(s) => push_separator(s, out),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Verbatim tier-text slicing
// ---------------------------------------------------------------------------

fn slice(input: &str, span: talkbank_model::Span) -> &str {
    input.get(span.to_range()).unwrap_or("")
}

/// Extract the content after `*CODE:` / `%tier:` from a raw tier line slice.
fn tier_content(raw: &str) -> String {
    raw.split_once(':')
        .map(|(_, rest)| rest.trim())
        .unwrap_or("")
        .to_string()
}

/// Split a raw dependent-tier line slice into (`%name`, content).
fn tier_name_and_content(raw: &str) -> Option<(String, String)> {
    let (name, content) = raw.split_once(':')?;
    Some((name.trim().to_string(), content.trim().to_string()))
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
        language: id.language.iter().next().map(|c| c.to_string()),
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
        parse_with_chatter(text, Some("%mor"), Some("%gra"), false)
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
        let a = parse_with_chatter(text, Some("%xmor"), Some("%xgra"), false);
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
}
