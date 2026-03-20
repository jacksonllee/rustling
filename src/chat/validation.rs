//! Validation for parsed CHAT files.
//!
//! The [`validate_chat_file`] function runs post-parse checks and returns
//! a list of [`ValidationError`]s.  When `strict=True` in the Python API,
//! any non-empty list causes a `ValueError`.

use crate::chat::header::{Headers, split_header_line};
use crate::chat::utterance::Utterance;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single validation error found in a CHAT file.
#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
}

impl ValidationError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

/// Valid CHAT participant roles.
static VALID_ROLES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "Target_Child",
        "Child",
        "Mother",
        "Father",
        "Brother",
        "Sister",
        "Sibling",
        "Grandmother",
        "Grandfather",
        "Relative",
        "Participant",
        "Investigator",
        "Partner",
        "Boy",
        "Girl",
        "Adult",
        "Teenager",
        "Male",
        "Female",
        "Visitor",
        "Friend",
        "Playmate",
        "Caretaker",
        "Environment",
        "Group",
        "Unidentified",
        "Uncertain",
        "Other",
        "Text",
        "Media",
        "PlayRole",
        "LENA",
        "Target_Adult",
        "Non_Human",
        "Nurse",
        "Doctor",
        "Babysitter",
        "Housekeeper",
        "Student",
        "Teacher",
        "Camera_Operator",
        "Observer",
        "Speaker",
        "Narrator",
        "Justice",
        "Attorney",
        "Offender",
        "Victim",
        "Witness",
        "Uncle",
        "Aunt",
        "Clinician",
        "Therapist",
        // Additional roles observed in valid CHAT corpora:
        "Teacher's_Aide",
        "Audience",
        "Guest",
        "Host",
        "Informant",
        "Leader",
        "Member",
        "Subject",
    ]
    .into_iter()
    .collect()
});

/// Valid @Options values.
static VALID_OPTIONS: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| ["CA", "CA-Unicode", "multi", "dummy"].into_iter().collect());

/// Known valid file-level headers (excluding changeable headers and
/// participant-specific headers like "Birth of X").
static VALID_FILE_HEADERS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "UTF8",
        "Begin",
        "End",
        "Window",
        "Font",
        "Color Words",
        "ColorWords",
        "PID",
        "Languages",
        "Participants",
        "Options",
        "ID",
        "Media",
        "Location",
        "Number",
        "Recording Quality",
        "Room Layout",
        "Tape Location",
        "Time Duration",
        "Time Start",
        "Transcriber",
        "Transcription",
        "Warning",
        // Changeable headers:
        "Activities",
        "Bck",
        "Bg",
        "Blank",
        "Comment",
        "Date",
        "Eg",
        "G",
        "New Episode",
        "Page",
        "Situation",
    ]
    .into_iter()
    .collect()
});

/// Valid form marker codes (after @).
static VALID_FORM_MARKERS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "s", "c", "d", "e", "f", "g", "i", "k", "l", "n", "o", "p", "q", "t", "u", "b", "wp", "si",
        "sl", "m", "fp", "fs", "ls", "sas", "z", "x", "w", "r", "h", "j", "v", "a",
    ]
    .into_iter()
    .collect()
});

// Regex for time marks in raw utterances (e.g., "123_456" at end of line).
static BULLET_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x15-?(\d+)_(\d+)-?\x15").unwrap());

// Regex for raw bullet-style timestamps at end of line (non-hidden).
static RAW_BULLET_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)_(\d+)\s*$").unwrap());

// Replacement [: word].
static REPLACEMENT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[:\s+([^\]]+)\]").unwrap());

// Fragment with replacement.
static FRAG_REPLACEMENT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"&~\S+\s+\[:\s+[^\]]+\]").unwrap());

// Adjacent quotes.
static ADJACENT_QUOTES_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""\s+""#).unwrap());

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Validate a parsed CHAT file and return any errors found.
///
/// This runs *after* parsing — the parser is lenient and always produces
/// a result; this function checks specification conformance.
pub fn validate_chat_file(
    file_path: &str,
    headers: &Headers,
    events: &[Utterance],
    raw_lines: &[String],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    validate_raw_structure(file_path, raw_lines, &mut errors);
    validate_headers(file_path, headers, raw_lines, &mut errors);
    validate_utterances(file_path, headers, events, raw_lines, &mut errors);
    errors
}

// ---------------------------------------------------------------------------
// 1. Raw-line structural checks
// ---------------------------------------------------------------------------

fn validate_raw_structure(
    file_path: &str,
    raw_lines: &[String],
    errors: &mut Vec<ValidationError>,
) {
    // Content after @End.
    let mut seen_end = false;
    for line in raw_lines {
        if line == "@End" {
            if seen_end {
                errors.push(ValidationError::new(format!(
                    "{file_path}: content after @End"
                )));
                break;
            }
            seen_end = true;
            continue;
        }
        if seen_end && (line.starts_with('*') || line.starts_with('%')) {
            errors.push(ValidationError::new(format!(
                "{file_path}: content after @End"
            )));
            break;
        }
    }

    // Duplicate dependent tiers within an utterance group.
    let mut current_dep_tiers: HashSet<String> = HashSet::new();
    for line in raw_lines {
        if line.starts_with('*') {
            current_dep_tiers.clear();
        } else if line.starts_with('%') {
            let tier_name = line.split(':').next().unwrap_or("");
            if !current_dep_tiers.insert(tier_name.to_string()) {
                errors.push(ValidationError::new(format!(
                    "{file_path}: duplicate dependent tier {tier_name}"
                )));
            }
        }
    }

    // Invalid/unknown file-level headers.
    for line in raw_lines {
        if !line.starts_with('@') {
            continue;
        }
        if line == "@End" || line == "@Begin" || line == "@UTF8" {
            continue;
        }

        let (name, _value) = split_header_line(line);

        // Skip participant-specific headers like "Birth of CHI".
        if name.contains(" of ") {
            continue;
        }

        if !VALID_FILE_HEADERS.contains(name)
            && !name.starts_with("Birth of")
            && !name.starts_with("Birthplace of")
            && !name.starts_with("L1 of")
        {
            // Known invalid headers:
            if name == "Exceptions" || name == "New Language" || name == "Code" {
                errors.push(ValidationError::new(format!(
                    "{file_path}: invalid header @{name}"
                )));
            }
        }
    }

    // @Page validation.
    for line in raw_lines {
        if !line.starts_with("@Page") {
            continue;
        }
        let (name, value) = split_header_line(line);
        if name == "Page" {
            match value {
                None => {
                    errors.push(ValidationError::new(format!(
                        "{file_path}: @Page header missing page number"
                    )));
                }
                Some(v) => {
                    if v.parse::<u64>().is_err() {
                        errors.push(ValidationError::new(format!(
                            "{file_path}: @Page header has non-numeric value: {v}"
                        )));
                    }
                }
            }
        }
    }

    // Missing @UTF8 header.
    if !raw_lines.is_empty() && raw_lines[0] != "@UTF8" && raw_lines[0] == "@Begin" {
        errors.push(ValidationError::new(format!(
            "{file_path}: file does not start with @UTF8"
        )));
    }
}

// ---------------------------------------------------------------------------
// 2. Header validations
// ---------------------------------------------------------------------------

fn validate_headers(
    file_path: &str,
    headers: &Headers,
    raw_lines: &[String],
    errors: &mut Vec<ValidationError>,
) {
    // @Options validation.
    if let Some(ref opts) = headers.options
        && !VALID_OPTIONS.contains(opts.as_str())
    {
        errors.push(ValidationError::new(format!(
            "{file_path}: invalid @Options value: {opts}"
        )));
    }

    // Participant code validation.
    for p in &headers.participants {
        if !p.code.is_ascii() {
            errors.push(ValidationError::new(format!(
                "{file_path}: non-ASCII participant code: {}",
                p.code
            )));
        }
        if p.code.contains('+') {
            errors.push(ValidationError::new(format!(
                "{file_path}: participant code contains '+': {}",
                p.code
            )));
        }
    }

    // Every participant in @Participants should have a corresponding @ID
    // (only check when at least one @ID line exists).
    let id_codes: HashSet<String> = raw_lines
        .iter()
        .filter(|l| l.starts_with("@ID:") || l.starts_with("@ID\t") || l.starts_with("@ID:\t"))
        .filter_map(|l| {
            let value = l.split_once('\t')?.1;
            let fields: Vec<&str> = value.split('|').collect();
            if fields.len() >= 3 {
                Some(fields[2].trim().to_string())
            } else {
                None
            }
        })
        .collect();
    if !id_codes.is_empty() {
        for p in &headers.participants {
            if !p.code.is_empty() && !id_codes.contains(&p.code) {
                errors.push(ValidationError::new(format!(
                    "{file_path}: participant {} is missing a required @ID header",
                    p.code
                )));
            }
        }
    }

    // Role validation.
    for p in &headers.participants {
        if !p.role.is_empty() {
            if !p.role.is_ascii() {
                errors.push(ValidationError::new(format!(
                    "{file_path}: invalid role (non-ASCII): {}",
                    p.role
                )));
            }
            if p.role.is_ascii() && !VALID_ROLES.contains(p.role.as_str()) {
                errors.push(ValidationError::new(format!(
                    "{file_path}: invalid role: {}",
                    p.role
                )));
            }
        }
    }

    // @ID role must match @Participants role.
    let participants_roles: HashMap<String, String> = headers
        .participants
        .iter()
        .filter(|p| !p.role.is_empty())
        .map(|p| (p.code.clone(), p.role.clone()))
        .collect();
    for line in raw_lines {
        if !(line.starts_with("@ID:") || line.starts_with("@ID\t") || line.starts_with("@ID:\t")) {
            continue;
        }
        if let Some(value) = line.split_once('\t').map(|x| x.1) {
            let fields: Vec<&str> = value.split('|').collect();
            if fields.len() >= 8 {
                let code = fields[2].trim();
                let id_role = fields[7].trim();
                if !id_role.is_empty()
                    && let Some(part_role) = participants_roles.get(code)
                    && !part_role.is_empty()
                    && part_role != id_role
                {
                    errors.push(ValidationError::new(format!(
                        "{file_path}: {code} has @ID role {id_role} but @Participants role {part_role}"
                    )));
                }
            }
        }
    }

    // Age format: months and days must be 2 digits.
    for line in raw_lines {
        if !(line.starts_with("@ID:") || line.starts_with("@ID\t") || line.starts_with("@ID:\t")) {
            continue;
        }
        if let Some(value) = line.split_once('\t').map(|x| x.1) {
            let fields: Vec<&str> = value.split('|').collect();
            if fields.len() >= 4 {
                let age_str = fields[3].trim();
                if !age_str.is_empty() {
                    validate_age_format(file_path, age_str, errors);
                }
            }
        }
    }

    // @Media filename must match CHAT file basename.
    if let Some(ref media) = headers.media_data {
        let file_basename = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        // Only check when file_path is a real .cha file (not a UUID from from_strs).
        if !media.filename.is_empty()
            && !file_basename.is_empty()
            && !media.filename.contains("://")
            && file_path.ends_with(".cha")
            && media.filename != file_basename
        {
            errors.push(ValidationError::new(format!(
                "{file_path}: @Media name '{}' does not match file name",
                media.filename
            )));
        }

        // Remote URLs in @Media must be quoted.
        for line in raw_lines {
            if line.starts_with("@Media") {
                let value = line.split_once('\t').map(|x| x.1).unwrap_or("");
                if value.contains("://") && !value.starts_with('"') {
                    errors.push(ValidationError::new(format!(
                        "{file_path}: @Media URL must be quoted"
                    )));
                }
                break;
            }
        }
    }

    // Invalid SES in @ID.
    for line in raw_lines {
        if !(line.starts_with("@ID:") || line.starts_with("@ID\t") || line.starts_with("@ID:\t")) {
            continue;
        }
        if let Some(value) = line.split_once('\t').map(|x| x.1) {
            let fields: Vec<&str> = value.split('|').collect();
            if fields.len() >= 7 {
                let ses_field = fields[6].trim();
                if !ses_field.is_empty() {
                    let valid_ses_patterns = [
                        "UC",
                        "MC",
                        "WC",
                        "LWC",
                        "LMC",
                        "UMC",
                        "upper-working-class",
                        "working-class",
                        "middle-class",
                    ];
                    let parts: Vec<&str> = ses_field.split(',').collect();
                    for part in &parts {
                        let part = part.trim();
                        if part.is_empty() {
                            continue;
                        }
                        if part.chars().all(|c| c.is_ascii_digit()) {
                            continue;
                        }
                        if valid_ses_patterns.contains(&part) {
                            continue;
                        }
                        if part.contains('-')
                            && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
                        {
                            continue;
                        }
                        errors.push(ValidationError::new(format!(
                            "{file_path}: invalid SES value in @ID: {ses_field}"
                        )));
                        break;
                    }
                }
            }
        }
    }

    // Invalid language code in @Languages.
    for lang in &headers.languages {
        if lang.len() == 3 && lang == "aaa" {
            errors.push(ValidationError::new(format!(
                "{file_path}: invalid language code: {lang}"
            )));
        }
    }

    // Obsolete dependent tier name.
    for line in raw_lines {
        if !line.starts_with('%') {
            continue;
        }
        if let Some(tier_name) = line.strip_prefix('%') {
            let tier_name = tier_name.split(':').next().unwrap_or("").trim();
            if tier_name == "lan" {
                errors.push(ValidationError::new(format!(
                    "{file_path}: obsolete dependent tier %{tier_name}"
                )));
            }
        }
    }
}

/// Validate age format from raw @ID string.
fn validate_age_format(file_path: &str, age_str: &str, errors: &mut Vec<ValidationError>) {
    let age_str = age_str.trim();
    if let Some((_years_str, rest)) = age_str.split_once(';') {
        if rest.is_empty() {
            return; // Just "N;" is OK.
        }
        let (months_str, days_str) = if let Some((m, d)) = rest.split_once('.') {
            (m, d)
        } else {
            (rest, "")
        };
        if !months_str.is_empty() && months_str.len() != 2 {
            errors.push(ValidationError::new(format!(
                "{file_path}: age months must be two digits, got: {age_str}"
            )));
        }
        if !days_str.is_empty() && days_str.len() != 2 {
            errors.push(ValidationError::new(format!(
                "{file_path}: age days must be two digits, got: {age_str}"
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Utterance checks
// ---------------------------------------------------------------------------

fn validate_utterances(
    file_path: &str,
    headers: &Headers,
    events: &[Utterance],
    raw_lines: &[String],
    errors: &mut Vec<ValidationError>,
) {
    let is_ca = headers
        .options
        .as_deref()
        .map(|o| o.starts_with("CA"))
        .unwrap_or(false);

    let mut utterance_raw_lines: Vec<&str> = Vec::new();
    for line in raw_lines {
        if line.starts_with('*') {
            utterance_raw_lines.push(line);
        }
    }

    for (idx, utt) in events.iter().enumerate() {
        if utt.changeable_header.is_some() {
            continue;
        }

        let raw_main = if let Some(ref tiers) = utt.tiers {
            if let Some(ref participant) = utt.participant {
                tiers.get(participant).map(|s| s.as_str()).unwrap_or("")
            } else {
                ""
            }
        } else {
            ""
        };

        let raw_line = utterance_raw_lines.get(idx).copied().unwrap_or("");
        let raw_text = raw_line
            .find(":\t")
            .map(|i| &raw_line[i + 2..])
            .unwrap_or(raw_main);

        validate_single_utterance(file_path, raw_text, is_ca, headers, errors);
    }

    // Bullet time validation: start must be < end.
    for line in raw_lines {
        if !line.starts_with('*') {
            continue;
        }
        for caps in BULLET_REGEX.captures_iter(line) {
            if let (Some(start_m), Some(end_m)) = (caps.get(1), caps.get(2))
                && let (Ok(start), Ok(end)) = (
                    start_m.as_str().parse::<i64>(),
                    end_m.as_str().parse::<i64>(),
                )
                && start >= end
            {
                errors.push(ValidationError::new(format!(
                    "{file_path}: bullet start {start} must be earlier than end {end}"
                )));
            }
        }
    }
}

/// Strip trailing post-code brackets like [+ IMP], [=! singing], etc.
fn strip_trailing_brackets(text: &str) -> &str {
    let mut end = text.len();
    let bytes = text.as_bytes();
    loop {
        while end > 0 && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t') {
            end -= 1;
        }
        if end == 0 {
            break;
        }
        if bytes[end - 1] == b']' {
            let mut depth = 1;
            let mut i = end - 2;
            loop {
                if bytes[i] == b']' {
                    depth += 1;
                } else if bytes[i] == b'[' {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                if i == 0 {
                    return &text[..end];
                }
                i -= 1;
            }
        } else {
            break;
        }
    }
    &text[..end]
}

fn validate_single_utterance(
    file_path: &str,
    raw_text: &str,
    is_ca: bool,
    headers: &Headers,
    errors: &mut Vec<ValidationError>,
) {
    let text = BULLET_REGEX.replace_all(raw_text, "");
    let text = RAW_BULLET_REGEX.replace_all(&text, "");
    let text = text.trim();

    if text.is_empty() {
        return;
    }

    let core_text = strip_trailing_brackets(text);
    let core_text = core_text.trim();
    if core_text.is_empty() {
        return;
    }

    // --- Terminator checks ---
    let terminators = ['.', '?', '!'];
    let last_char = core_text.chars().last().unwrap_or(' ');

    if !is_ca {
        // Missing terminator.
        if !terminators.contains(&last_char)
            && !core_text.ends_with("+...")
            && !core_text.ends_with("+/.")
            && !core_text.ends_with("+//.")
            && !core_text.ends_with("+/?")
            && !core_text.ends_with("+//?")
            && !core_text.ends_with("+\"/.")
        {
            errors.push(ValidationError::new(format!(
                "{file_path}: utterance missing terminator"
            )));
            return;
        }

        // Multiple terminators (e.g., "!?").
        if core_text.len() >= 2 {
            let chars: Vec<char> = core_text.chars().collect();
            let len = chars.len();
            let second_last = chars[len - 2];
            if terminators.contains(&last_char)
                && terminators.contains(&second_last)
                && !(second_last == '.' && last_char == '.' && len >= 3 && chars[len - 3] == '.')
                && !core_text.ends_with("+...")
                && !core_text.ends_with("+..?")
            {
                errors.push(ValidationError::new(format!(
                    "{file_path}: multiple terminators"
                )));
            }
        }

        // CA-style terminator in non-CA transcript.
        if core_text.ends_with("+=.") || core_text.ends_with("+=?") || core_text.ends_with("+=!") {
            errors.push(ValidationError::new(format!(
                "{file_path}: CA-style terminator in non-CA transcript"
            )));
        }

        // Tone terminators.
        if core_text.len() >= 2 {
            let bytes = core_text.as_bytes();
            let len = bytes.len();
            if bytes[len - 2] == b'-'
                && (last_char == '.' || last_char == '!' || last_char == '?' || last_char == '\'')
                && (len == 2 || bytes[len - 3] == b' ')
            {
                errors.push(ValidationError::new(format!(
                    "{file_path}: tone terminator not allowed"
                )));
            }
            if len >= 3
                && core_text.ends_with(",.")
                && bytes[len - 3] == b'-'
                && (len == 3 || bytes[len - 4] == b' ')
            {
                errors.push(ValidationError::new(format!(
                    "{file_path}: tone terminator not allowed"
                )));
            }
        }
    }

    // --- Empty utterance check ---
    let mut content_text = core_text;
    content_text = content_text.trim_end_matches(|c: char| terminators.contains(&c) || c == ' ');
    for suffix in &["+...", "+/.", "+//.", "+/?", "+//?", "+\"/.", "+..?"] {
        if let Some(stripped) = content_text.strip_suffix(suffix) {
            content_text = stripped;
            break;
        }
    }
    content_text = content_text.trim();
    if content_text.is_empty() {
        errors.push(ValidationError::new(format!(
            "{file_path}: empty utterance (no content before terminator)"
        )));
        return;
    }

    // --- Word-level checks ---
    let words = extract_words(text);

    for word in &words {
        // Uppercase unintelligible markers.
        if *word == "XXX" || *word == "YYY" || *word == "WWW" {
            errors.push(ValidationError::new(format!(
                "{file_path}: \"{word}\" must be lowercase"
            )));
        }

        // Obsolete unintelligible markers.
        if *word == "xx" || *word == "yy" {
            errors.push(ValidationError::new(format!(
                "{file_path}: \"{word}\" is obsolete; use \"{}\"",
                if *word == "xx" { "xxx" } else { "yyy" }
            )));
        }

        // ^ at beginning of word (old blocking).
        if word.starts_with('^') {
            errors.push(ValidationError::new(format!(
                "{file_path}: ^ not allowed at beginning of word: {word}"
            )));
        }

        // &=0 pattern (illegal).
        if word.starts_with("&=0") {
            errors.push(ValidationError::new(format!(
                "{file_path}: &=0 notation is illegal: {word}"
            )));
        }

        // &= with fraction (e.g., "&=1.88").
        if let Some(rest) = word.strip_prefix("&=")
            && rest.contains('.')
            && rest
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        {
            errors.push(ValidationError::new(format!(
                "{file_path}: invalid &= notation (fraction): {word}"
            )));
        }

        // Old & notation.
        if word.starts_with('&')
            && !word.starts_with("&=")
            && !word.starts_with("&~")
            && !word.starts_with("&-")
            && !word.starts_with("&+")
            && !word.starts_with("&*")
            && word.len() > 1
        {
            let rest = &word[1..];
            if rest
                .chars()
                .next()
                .map(|c| c.is_alphabetic())
                .unwrap_or(false)
                && !rest.starts_with("l")
            {
                errors.push(ValidationError::new(format!(
                    "{file_path}: old & notation not allowed: {word}"
                )));
            }
        }

        // Illegal word with 0-affix (e.g., "Ollie-0's").
        if word.contains("-0") && !word.starts_with("0") && !word.starts_with("&") {
            errors.push(ValidationError::new(format!(
                "{file_path}: illegal 0-affix in word: {word}"
            )));
        }

        // Pause embedded in word.
        if word.contains("(.)") && *word != "(.)" {
            errors.push(ValidationError::new(format!(
                "{file_path}: pause marker embedded in word: {word}"
            )));
        }
        if word.contains("(..)") && *word != "(..)" {
            errors.push(ValidationError::new(format!(
                "{file_path}: pause marker embedded in word: {word}"
            )));
        }
        if word.contains("(...)") && *word != "(...)" {
            errors.push(ValidationError::new(format!(
                "{file_path}: pause marker embedded in word: {word}"
            )));
        }

        // Word entirely in parentheses (no spoken content).
        if !is_ca
            && word.starts_with('(')
            && word.ends_with(')')
            && !word.starts_with("(.)")
            && !word.starts_with("(..)")
            && !word.starts_with("(...)")
            && word.len() > 2
        {
            let inner = &word[1..word.len() - 1];
            let is_single_group = !inner.contains('(') && !inner.contains(')');
            if is_single_group {
                let is_timed_pause = !inner.is_empty()
                    && inner
                        .chars()
                        .all(|c| c.is_ascii_digit() || c == '.' || c == ':');
                if !is_timed_pause {
                    errors.push(ValidationError::new(format!(
                        "{file_path}: word has no spoken content (entirely parenthesized): {word}"
                    )));
                }
            }
        }

        // Form marker validation.
        validate_form_markers(file_path, word, headers, errors);

        // @q suffix -s check.
        if word.contains("@q-") || word.contains("@ap-") {
            errors.push(ValidationError::new(format!(
                "{file_path}: extraneous suffix after form marker: {word}"
            )));
        }
    }

    // --- Inline dependent tier brackets ---
    let inline_dep_tiers: &[&str] = &["[%act:", "[%sch:", "[%sdi:", "[%ssx:"];
    for pattern in inline_dep_tiers {
        if text.contains(pattern) {
            errors.push(ValidationError::new(format!(
                "{file_path}: inline dependent tier in utterance: {pattern}"
            )));
        }
    }

    // --- Quotation checks ---
    validate_quotations(file_path, text, errors);

    // --- Replacement checks ---
    validate_replacements(file_path, text, errors);
}

/// Extract words from utterance text, skipping brackets and annotations.
fn extract_words(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();

    while i < len {
        if bytes[i] == b' ' || bytes[i] == b'\t' {
            i += 1;
            continue;
        }

        // Skip bracket groups.
        if bytes[i] == b'[' {
            let mut depth = 1;
            i += 1;
            while i < len && depth > 0 {
                if bytes[i] == b'[' {
                    depth += 1;
                } else if bytes[i] == b']' {
                    depth -= 1;
                }
                i += 1;
            }
            continue;
        }

        // Skip angle-bracket groups (but extract words within).
        if bytes[i] == b'<' {
            let start = i;
            let mut depth = 1;
            i += 1;
            while i < len && depth > 0 {
                if bytes[i] == b'<' {
                    depth += 1;
                } else if bytes[i] == b'>' {
                    depth -= 1;
                }
                i += 1;
            }
            if i <= len {
                let inner = &text[start + 1..i.saturating_sub(1)];
                words.extend(extract_words(inner));
            }
            continue;
        }

        // Collect a word.
        let start = i;
        while i < len && bytes[i] != b' ' && bytes[i] != b'\t' {
            if bytes[i] == b'[' {
                break;
            }
            i += 1;
        }
        if i > start {
            let word = &text[start..i];
            if !word.contains('\x15') {
                words.push(word);
            }
        }
    }

    words
}

fn validate_form_markers(
    file_path: &str,
    word: &str,
    headers: &Headers,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(at_pos) = word.find('@') {
        if at_pos == 0 {
            return;
        }

        let marker = &word[at_pos + 1..];
        let marker =
            marker.trim_end_matches(|c: char| ".?!\"'\u{201C}\u{201D}\u{2018}\u{2019}".contains(c));
        let marker_code = marker.split(':').next().unwrap_or(marker);
        let marker_code = marker_code.split('$').next().unwrap_or(marker_code);
        let marker_base = marker_code.split('-').next().unwrap_or(marker_code);

        if marker_base.is_empty() {
            return;
        }

        // Invalid form marker code.
        if !VALID_FORM_MARKERS.contains(marker_base) {
            errors.push(ValidationError::new(format!(
                "{file_path}: invalid form marker code: @{marker_base} in word: {word}"
            )));
        }

        // Compound word with form marker.
        let word_before_at = &word[..at_pos];
        let is_compound = word_before_at.contains('+');
        let is_sign_mode = headers
            .options
            .as_deref()
            .map(|o| o == "sign")
            .unwrap_or(false);
        if is_compound && !is_sign_mode && (marker_base == "s" || marker_base == "o") {
            errors.push(ValidationError::new(format!(
                "{file_path}: compound word may not have form marker @{marker_base}: {word}"
            )));
        }

        // @l must be a single letter.
        let bare_word = word_before_at
            .trim_start_matches(|c: char| "\"\u{201C}\u{201D}\u{2018}\u{2019}".contains(c));
        if marker_base == "l" && bare_word.chars().count() != 1 {
            errors.push(ValidationError::new(format!(
                "{file_path}: @l form marker requires a single letter: {word}"
            )));
        }
    }
}

fn validate_quotations(file_path: &str, text: &str, errors: &mut Vec<ValidationError>) {
    // Unicode curly quotes not allowed.
    if text.contains('\u{2018}') || text.contains('\u{2019}') {
        errors.push(ValidationError::new(format!(
            "{file_path}: Unicode curly quotes not allowed in utterance"
        )));
        return;
    }

    // Multiple quotation sections (adjacent quotes).
    let cleaned = text.replace("+\"", "").replace("+\"/.", "");
    let quote_count = cleaned.chars().filter(|&c| c == '"').count();
    if quote_count > 2 && ADJACENT_QUOTES_REGEX.is_match(&cleaned) {
        errors.push(ValidationError::new(format!(
            "{file_path}: multiple quotation sections in utterance"
        )));
    }
}

fn validate_replacements(file_path: &str, text: &str, errors: &mut Vec<ValidationError>) {
    for caps in REPLACEMENT_REGEX.captures_iter(text) {
        let replacement = caps.get(1).unwrap().as_str().trim();

        // Replacement cannot be untranscribed.
        if replacement == "xxx" || replacement == "yyy" || replacement == "www" {
            errors.push(ValidationError::new(format!(
                "{file_path}: replacement word cannot be untranscribed: [: {replacement}]"
            )));
        }

        // Replacement cannot be an omission (0-prefixed).
        if replacement.starts_with('0') && replacement.len() > 1 {
            errors.push(ValidationError::new(format!(
                "{file_path}: replacement word cannot be an omission: [: {replacement}]"
            )));
        }
    }

    // Fragment with replacement.
    if FRAG_REPLACEMENT_REGEX.is_match(text) {
        errors.push(ValidationError::new(format!(
            "{file_path}: replacement not allowed for a fragment"
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_words_basic() {
        assert_eq!(extract_words("hello world ."), vec!["hello", "world", "."]);
    }

    #[test]
    fn test_extract_words_skips_brackets() {
        assert_eq!(
            extract_words("aam [: m_d@s] [*] player@s ."),
            vec!["aam", "player@s", "."]
        );
    }

    #[test]
    fn test_extract_words_angle_group() {
        assert_eq!(
            extract_words("<word1 word2> [/] word3 ."),
            vec!["word1", "word2", "word3", "."]
        );
    }

    #[test]
    fn test_extract_words_angle_group_with_brackets() {
        assert_eq!(
            extract_words("<aam [: m_d@s] [*]> [/] aam [: m_d@s] [*] player@s ."),
            vec!["aam", "aam", "player@s", "."]
        );
    }
}
