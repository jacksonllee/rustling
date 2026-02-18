//! Utterance cleaning for CHAT transcription data.
//!
//! Uses a two-pass bracket-aware parser:
//! 1. **Tokenize**: scan the utterance left-to-right, recognizing brackets (`[…]`),
//!    angle groups (`<…>`), timestamps, and timed pauses as structural segments.
//! 2. **Process**: walk the segment stream to handle drops, replacements,
//!    retracings, and word-level filtering.

use std::collections::HashSet;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Segment types produced by the tokenizer
// ---------------------------------------------------------------------------

enum Segment {
    /// A regular word or punctuation token.
    Word(String),
    /// An angle-bracketed group `<word1 word2>` (used for multi-word scoping).
    AngleGroup(Vec<String>),
    /// Any annotation bracket that should be silently dropped.
    Drop,
    /// `[: replacement]` — replace the preceding Word/AngleGroup with these words.
    Replace(Vec<String>),
    /// `[:: …]` — keep the preceding Word/AngleGroup, discard this bracket.
    KeepOriginal,
    /// `[/]`, `[//]`, `[///]`, `[/?]`, `[/-]` — drop the preceding Word/AngleGroup.
    Retracing,
}

/// Intermediate output during the processing pass.
enum OutputItem {
    Word(String),
    Group(Vec<String>),
}

// ---------------------------------------------------------------------------
// Static data for word filtering (Stage 5 — kept from original)
// ---------------------------------------------------------------------------

static ESCAPE_PREFIXES: &[&str] = &[
    "[?", "[/", "[<", "[>", "[:", "[!", "[*", "+\"", "+,", "<&", "&",
];

static ESCAPE_SUFFIXES: &[&str] = &["\u{21ab}xxx"]; // ↫xxx

static ESCAPE_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "0",
        "++",
        "+<",
        "+^",
        "(.)",
        "(..)",
        "(...)",
        ":",
        ";",
        ";;",
        "<",
        ">",
        "xx",
        "yy",
        "xxx",
        "yyy",
        "www",
        "www:",
        "xxx:",
        "xxx;",
        "xxx;;",
        "xxx\u{2192}", // xxx→
        "xxx\u{2191}", // xxx↑
        "xxx@si",
        "yyy:",
        "\u{2192}", // →
    ]
    .into_iter()
    .collect()
});

static KEEP_PREFIXES: &[&str] = &["+\"/", "+,/", "+\"."];

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Flush the accumulated word buffer into the segment list.
fn flush_word(buf: &mut String, segments: &mut Vec<Segment>) {
    if buf.is_empty() {
        return;
    }
    let word = std::mem::take(buf);
    segments.push(Segment::Word(word));
}

/// Check whether `<` at position `pos` should start an angle group.
/// Returns `true` when we are at a word boundary (empty buffer) and there is a
/// matching `>` somewhere ahead.
fn should_start_angle_group(chars: &[char], pos: usize, word_buf: &str) -> bool {
    if !word_buf.is_empty() {
        return false;
    }
    // Ensure there is a matching '>'.
    chars[pos + 1..].contains(&'>')
}

/// Classify the text between `[` and `]` into a [`Segment`].
fn classify_bracket(content: &str) -> Segment {
    // Standalone codes (exact match after trimming).
    match content.trim() {
        "/" | "//" | "///" | "/?" | "/-" | "e" => return Segment::Retracing,
        "?" | "!" | "!!" | "^c" | "*" => return Segment::Drop,
        _ => {}
    }

    // Overlap markers: [<], [>], [<N], [>N].
    let trimmed = content.trim();
    if let Some(rest) = trimmed
        .strip_prefix('<')
        .or_else(|| trimmed.strip_prefix('>'))
        && (rest.is_empty() || rest.chars().all(|c| c.is_ascii_digit()))
    {
        return Segment::Drop;
    }

    // Replacement brackets (order matters: check `::` before `:`).
    if let Some(rest) = content.strip_prefix(":: ") {
        // [:: replacement] — keep original, drop this.
        let _ = rest; // content is unused; we just keep the preceding element.
        return Segment::KeepOriginal;
    }
    if let Some(rest) = content.strip_prefix(": ") {
        let words: Vec<String> = rest.split_whitespace().map(String::from).collect();
        return Segment::Replace(words);
    }

    // Drop patterns: [= …], [+ …], [* …], [% …], [- …], [^ …], [# …],
    // [=? …], [=! …], [x N], [%act: …].
    if content.starts_with("= ")
        || content.starts_with("=? ")
        || content.starts_with("=! ")
        || content.starts_with("+ ")
        || content.starts_with("* ")
        || content.starts_with("% ")
        || content.starts_with("- ")
        || content.starts_with("^ ")
        || content.starts_with("# ")
        || content.starts_with("x ")
        || content.starts_with("%act: ")
    {
        return Segment::Drop;
    }

    // Unrecognized bracket — keep as a word token (preserves original text).
    Segment::Word(format!("[{content}]"))
}

/// Return `true` when `content` (the text between `(` and `)`) is a timed
/// pause such as `1.5`, `2:30`, or `0:01.23`.
fn is_timed_pause(content: &str) -> bool {
    let b = content.as_bytes();
    if b.is_empty() {
        return false;
    }
    let mut i = 0;

    // Optional "digits:" prefix.
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b':' {
        if i == start {
            return false;
        }
        i += 1;
    } else {
        i = start;
    }

    // Required: at least one digit.
    let digit_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start {
        return false;
    }

    // Optional fractional part ".digits".
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }

    i == b.len()
}

/// Tokenize a CHAT utterance string into structural segments.
fn tokenize(input: &str) -> Vec<Segment> {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut segments: Vec<Segment> = Vec::new();
    let mut i = 0;
    let mut word_buf = String::new();

    while i < len {
        match chars[i] {
            // Timestamp: \x15…\x15 — drop entirely.
            '\x15' => {
                flush_word(&mut word_buf, &mut segments);
                i += 1;
                while i < len && chars[i] != '\x15' {
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
            }

            // Square bracket: classify annotation.
            '[' => {
                flush_word(&mut word_buf, &mut segments);
                i += 1;
                let mut content = String::new();
                while i < len && chars[i] != ']' {
                    content.push(chars[i]);
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
                segments.push(classify_bracket(&content));
            }

            // Angle bracket: multi-word scoping group.
            // Recursively tokenize/process the inner content so that brackets,
            // Unicode chars, timestamps, and pauses inside are handled.
            '<' if should_start_angle_group(&chars, i, &word_buf) => {
                flush_word(&mut word_buf, &mut segments);
                i += 1;
                let mut content = String::new();
                let mut depth: usize = 1;
                while i < len && depth > 0 {
                    match chars[i] {
                        // Skip bracket content so that <, > inside [...] don't
                        // affect nesting depth (e.g. [<], [>]).
                        '[' => {
                            content.push('[');
                            i += 1;
                            while i < len && chars[i] != ']' {
                                content.push(chars[i]);
                                i += 1;
                            }
                            if i < len {
                                content.push(']');
                                i += 1;
                            }
                        }
                        '<' => {
                            depth += 1;
                            content.push('<');
                            i += 1;
                        }
                        '>' => {
                            depth -= 1;
                            if depth > 0 {
                                content.push('>');
                            }
                            i += 1;
                        }
                        ch => {
                            content.push(ch);
                            i += 1;
                        }
                    }
                }
                let inner_segments = tokenize(&content);
                let words = process(&inner_segments);
                if !words.is_empty() {
                    segments.push(Segment::AngleGroup(words));
                }
            }

            // Parenthesized content: timed pause `(1.5)` is dropped.
            '(' => {
                let mut j = i + 1;
                while j < len && chars[j] != ')' {
                    j += 1;
                }
                if j < len {
                    let content: String = chars[i + 1..j].iter().collect();
                    if is_timed_pause(&content) {
                        flush_word(&mut word_buf, &mut segments);
                        i = j + 1;
                    } else {
                        word_buf.push('(');
                        i += 1;
                    }
                } else {
                    word_buf.push('(');
                    i += 1;
                }
            }

            // Unicode characters to skip (do NOT flush — these can appear
            // mid-word, e.g. `þa⌈ð` should become `það`).
            '\u{2039}' | '\u{203a}' // ‹ › guillemets
            | '\u{2308}' | '\u{2309}' // ⌈ ⌉ overlap
            | '\u{230a}' | '\u{230b}' // ⌊ ⌋ overlap
            | '\u{201c}' | '\u{201d}' // " " curly quotes
            => {
                i += 1;
            }

            // Comma: separate from the preceding word (so escape-word
            // filtering works on `xx,` → `xx` + `,`), but keep `+,` intact
            // because it is a CHAT linker prefix.
            ',' => {
                if word_buf == "+" {
                    word_buf.push(',');
                } else {
                    flush_word(&mut word_buf, &mut segments);
                    segments.push(Segment::Word(",".to_string()));
                }
                i += 1;
            }

            // Whitespace.
            ' ' | '\t' | '\n' | '\r' => {
                flush_word(&mut word_buf, &mut segments);
                i += 1;
            }

            // Any other character.
            ch => {
                word_buf.push(ch);
                i += 1;
            }
        }
    }
    flush_word(&mut word_buf, &mut segments);
    segments
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Walk the segment stream, applying retracings, replacements and drops.
fn process(segments: &[Segment]) -> Vec<String> {
    let mut output: Vec<OutputItem> = Vec::new();
    let mut i = 0;

    while i < segments.len() {
        match &segments[i] {
            Segment::Word(w) => {
                output.push(OutputItem::Word(w.clone()));
            }
            Segment::AngleGroup(words) => {
                output.push(OutputItem::Group(words.clone()));
            }
            Segment::Drop | Segment::KeepOriginal => {
                // Silently skip.
            }
            Segment::Replace(replacement) => {
                // Replace the most recent Word/AngleGroup.
                if let Some(pos) = output
                    .iter()
                    .rposition(|item| matches!(item, OutputItem::Word(_) | OutputItem::Group(_)))
                {
                    output[pos] = OutputItem::Group(replacement.clone());
                }
            }
            Segment::Retracing => {
                // Consecutive retracings collapse to a single one.
                while i + 1 < segments.len() && matches!(&segments[i + 1], Segment::Retracing) {
                    i += 1;
                }
                // Remove the most recent Word/AngleGroup.
                if let Some(pos) = output
                    .iter()
                    .rposition(|item| matches!(item, OutputItem::Word(_) | OutputItem::Group(_)))
                {
                    output.remove(pos);
                }
            }
        }
        i += 1;
    }

    // Flatten to a word list.
    let mut words = Vec::new();
    for item in output {
        match item {
            OutputItem::Word(w) => words.push(w),
            OutputItem::Group(ws) => words.extend(ws),
        }
    }
    words
}

// ---------------------------------------------------------------------------
// Word filtering
// ---------------------------------------------------------------------------

/// Clean residual angle/bracket characters from word boundaries.
fn clean_word_boundaries(word: &str) -> &str {
    let mut w = word;
    if let Some(rest) = w.strip_prefix('<') {
        w = rest;
    }
    if let Some(rest) = w.strip_suffix('>') {
        w = rest;
    }
    if let Some(rest) = w.strip_suffix(']') {
        w = rest;
    }
    w
}

/// Filter and clean individual words (escape words, fillers, etc.).
fn filter_words(words: Vec<String>) -> Vec<String> {
    let mut result = Vec::new();
    for raw in words {
        let word = clean_word_boundaries(&raw);
        if word.is_empty() {
            continue;
        }
        // Keep certain prefixed words.
        if KEEP_PREFIXES.iter().any(|k| word.starts_with(k)) {
            result.push(word.to_string());
            continue;
        }
        // Filter out omitted words (0-prefixed, e.g., 0you, 0the, 0學).
        if word.starts_with('0') && word[1..].starts_with(|c: char| c.is_alphabetic()) {
            continue;
        }
        // Filter out escape words, prefixes, and suffixes.
        if !ESCAPE_WORDS.contains(word)
            && !ESCAPE_PREFIXES.iter().any(|e| word.starts_with(e))
            && !ESCAPE_SUFFIXES.iter().any(|e| word.ends_with(e))
        {
            result.push(word.to_string());
        }
    }
    result
}

/// Split a trailing sentence-final period or question mark from the last word.
/// Handles cases like `"cookie."` → `["cookie", "."]` and `"what?"` → `["what", "?"]`.
fn split_trailing_punct(words: &mut Vec<String>) {
    if let Some(last) = words.last()
        && last.len() > 1
    {
        let bytes = last.as_bytes();
        let final_byte = bytes[bytes.len() - 1];
        let penult_byte = bytes[bytes.len() - 2];
        if (final_byte == b'.' || final_byte == b'?') && penult_byte.is_ascii_lowercase() {
            let word = last[..last.len() - 1].to_string();
            let punct = last[last.len() - 1..].to_string();
            let len = words.len();
            words[len - 1] = word;
            words.push(punct);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Clean a CHAT utterance by removing annotations and normalizing text.
pub(crate) fn clean_utterance(utterance: &str) -> String {
    let segments = tokenize(utterance);
    let words = process(&segments);
    let mut words = filter_words(words);
    split_trailing_punct(&mut words);
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        assert_eq!(clean_utterance(""), "");
    }

    #[test]
    fn test_simple_utterance() {
        assert_eq!(clean_utterance("I want cookie ."), "I want cookie .");
    }

    #[test]
    fn test_drop_explanation() {
        assert_eq!(
            clean_utterance("I want [= desire] cookie ."),
            "I want cookie ."
        );
    }

    #[test]
    fn test_drop_repetition_count() {
        assert_eq!(clean_utterance("cookie [x 3] ."), "cookie .");
    }

    #[test]
    fn test_drop_actions() {
        assert_eq!(clean_utterance("hello [+ IMP] ."), "hello .");
    }

    #[test]
    fn test_drop_error_marker() {
        assert_eq!(clean_utterance("goed [*] ."), "goed .");
    }

    #[test]
    fn test_drop_overlap_markers() {
        assert_eq!(clean_utterance("hello [<] world ."), "hello world .");
        assert_eq!(clean_utterance("hello [>] world ."), "hello world .");
    }

    #[test]
    fn test_drop_pauses() {
        assert_eq!(clean_utterance("hello (1.5) world ."), "hello world .");
    }

    #[test]
    fn test_timestamp_removal() {
        let input = "hello \x15123_456\x15 .";
        assert_eq!(clean_utterance(input), "hello .");
    }

    #[test]
    fn test_reformulation_single_word() {
        assert_eq!(clean_utterance("dog [//] cat ."), "cat .");
    }

    #[test]
    fn test_repetition_single_word() {
        assert_eq!(clean_utterance("the [/] the dog ."), "the dog .");
    }

    #[test]
    fn test_reformulation_multi_word() {
        assert_eq!(clean_utterance("< the dog > [//] the cat ."), "the cat .");
    }

    #[test]
    fn test_escape_words_removed() {
        assert_eq!(clean_utterance("xxx ."), ".");
        assert_eq!(clean_utterance("yyy ."), ".");
        assert_eq!(clean_utterance("www ."), ".");
    }

    #[test]
    fn test_filler_removed() {
        // & prefixed words are escape-prefixed
        assert_eq!(clean_utterance("&um hello ."), "hello .");
    }

    #[test]
    fn test_curly_quotes_removed() {
        assert_eq!(clean_utterance("\u{201c}hello\u{201d} ."), "hello .");
    }

    #[test]
    fn test_question_mark_spacing() {
        assert_eq!(clean_utterance("what ?"), "what ?");
    }

    #[test]
    fn test_sentence_final_period_spacing() {
        assert_eq!(clean_utterance("cookie."), "cookie .");
    }

    #[test]
    fn test_correction_keep_original() {
        // [:: ...] means keep original, drop correction
        assert_eq!(clean_utterance("goed [:: went] ."), "goed .");
    }

    #[test]
    fn test_correction_use_replacement() {
        // [: ...] means use replacement
        assert_eq!(clean_utterance("goed [: went] ."), "went .");
    }

    #[test]
    fn test_unicode_brackets_removed() {
        assert_eq!(clean_utterance("\u{2308}hello\u{2309} ."), "hello .");
    }

    // New test cases --------------------------------------------------------

    #[test]
    fn test_question_mark_attached_to_word() {
        // Bug fix: the old regex ate the char before '?'.
        assert_eq!(clean_utterance("what?"), "what ?");
    }

    #[test]
    fn test_nested_reformulations() {
        // Two consecutive retracings collapse, only the preceding group is dropped.
        assert_eq!(clean_utterance("< a b > [//] [/] the cat ."), "the cat .");
    }

    #[test]
    fn test_multi_word_replacement() {
        assert_eq!(clean_utterance("goed [: had gone] ."), "had gone .");
    }

    #[test]
    fn test_angle_group_replacement() {
        assert_eq!(clean_utterance("< the dog > [: the cat] ."), "the cat .");
    }

    #[test]
    fn test_error_marker_before_retracing() {
        assert_eq!(clean_utterance("word [*] [//] next ."), "next .");
    }

    #[test]
    fn test_multiple_annotations() {
        assert_eq!(
            clean_utterance("hello [= greeting] [+ IMP] world ."),
            "hello world ."
        );
    }

    #[test]
    fn test_uncertain_explanation() {
        assert_eq!(clean_utterance("word [=? maybe this] ."), "word .");
    }

    #[test]
    fn test_paralinguistic() {
        assert_eq!(clean_utterance("hello [=! laughing] ."), "hello .");
    }

    #[test]
    fn test_precode() {
        assert_eq!(clean_utterance("[- eng] hello ."), "hello .");
    }

    #[test]
    fn test_pause_dots_filtered() {
        assert_eq!(clean_utterance("hello (.) world ."), "hello world .");
        assert_eq!(clean_utterance("hello (..) world ."), "hello world .");
        assert_eq!(clean_utterance("hello (...) world ."), "hello world .");
    }

    #[test]
    fn test_timed_pause_with_colon() {
        assert_eq!(clean_utterance("hello (2:30.5) world ."), "hello world .");
    }

    #[test]
    fn test_false_start() {
        // [/-] drops the immediately preceding word only.
        assert_eq!(
            clean_utterance("want [/-] I need cookie ."),
            "I need cookie ."
        );
    }

    #[test]
    fn test_false_start_angle_group() {
        // Use angle brackets to scope multiple words.
        assert_eq!(
            clean_utterance("< I want > [/-] I need cookie ."),
            "I need cookie ."
        );
    }

    #[test]
    fn test_completion() {
        assert_eq!(clean_utterance("I [///] she went ."), "she went .");
    }

    #[test]
    fn test_omitted_words_filtered() {
        // 0-prefixed words are omitted words — no %mor entry.
        assert_eq!(clean_utterance("0you go ."), "go .");
        assert_eq!(clean_utterance("I 0can go ."), "I go .");
        assert_eq!(clean_utterance("0the dog ."), "dog .");
        assert_eq!(
            clean_utterance("I going 0to do another Bx ."),
            "I going do another Bx ."
        );
        // Non-ASCII 0-prefixed words should also be filtered.
        assert_eq!(clean_utterance("0學 去 ."), "去 .");
        assert_eq!(clean_utterance("0你 好 ."), "好 .");
        // Standalone "0" is also filtered (already covered by ESCAPE_WORDS).
        assert_eq!(clean_utterance("0 dog ."), "dog .");
    }

    #[test]
    fn test_nested_angle_brackets_retracing() {
        // Outer <...> scopes an inner <word> plus annotation for [//].
        assert_eq!(
            clean_utterance("<<how'd> [=? how]> [//] (.) how you hafta do the man ?"),
            "how you hafta do the man ?"
        );
    }

    #[test]
    fn test_nested_angle_brackets_repetition() {
        // Outer <...> scopes an inner <words> plus overlap marker for [/].
        assert_eq!(
            clean_utterance(
                "<<I got> [<]> [/] I got ink on my fingers <and> [/] and shoe polish ."
            ),
            "I got ink on my fingers and shoe polish ."
        );
    }

    #[test]
    fn test_exclude_single_word() {
        assert_eq!(
            clean_utterance("this is a mor [e] exclude ."),
            "this is a exclude ."
        );
    }

    #[test]
    fn test_exclude_angle_group() {
        assert_eq!(
            clean_utterance("this is <a multi-word> [e] exclude ."),
            "this is exclude ."
        );
    }

    #[test]
    fn test_is_timed_pause() {
        assert!(is_timed_pause("1"));
        assert!(is_timed_pause("1.5"));
        assert!(is_timed_pause("2:30"));
        assert!(is_timed_pause("2:30.5"));
        assert!(is_timed_pause("0:01.23"));
        assert!(!is_timed_pause(""));
        assert!(!is_timed_pause("."));
        assert!(!is_timed_pause(".."));
        assert!(!is_timed_pause("..."));
        assert!(!is_timed_pause("abc"));
        assert!(!is_timed_pause(":5"));
    }
}
