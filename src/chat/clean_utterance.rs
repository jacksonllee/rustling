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
    /// Any annotation bracket that should be silently dropped, keeping whatever
    /// preceded it (this covers replacements such as `[: …]` and `[:: …]`,
    /// where the audible view keeps the original word).
    Drop,
    /// `[x N]` -- repeat the preceding Word/AngleGroup N times total.
    Expand(usize),
}

/// Intermediate output during the processing pass.
#[derive(Clone)]
enum OutputItem {
    Word(String),
    Group(Vec<String>),
}

// ---------------------------------------------------------------------------
// Static data for word filtering (Stage 5 — kept from original)
// ---------------------------------------------------------------------------

static ESCAPE_SUFFIXES: &[&str] = &["\u{21ab}xxx"]; // ↫xxx

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
///
/// Retracings become no-ops (preceding material is kept), replacements keep the
/// original word, and `[x N]` expands repetitions -- the audibly faithful view.
fn classify_bracket(content: &str) -> Segment {
    // Standalone codes (exact match after trimming).
    match content.trim() {
        // Retracings keep the preceding material -- just drop the bracket.
        "/" | "//" | "///" | "/?" | "/-" | "e" => return Segment::Drop,
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

    // Replacement brackets: `[: …]` and `[:: …]` keep the original word and
    // discard the replacement, which is just dropping the bracket.
    if content.starts_with(":: ") || content.starts_with(": ") {
        return Segment::Drop;
    }

    // Drop patterns: [= …], [+ …], [* …], [% …], [- …], [^ …], [# …],
    // [=? …], [=! …], [%act: …].
    if content.starts_with("= ")
        || content.starts_with("=? ")
        || content.starts_with("=! ")
        || content.starts_with("+ ")
        || content.starts_with("* ")
        || content.starts_with("% ")
        || content.starts_with("- ")
        || content.starts_with("^ ")
        || content.starts_with("# ")
        || content.starts_with("%act: ")
    {
        return Segment::Drop;
    }

    // [x N] -- repetition count.
    if let Some(rest) = content.strip_prefix("x ") {
        if let Ok(n) = rest.trim().parse::<usize>() {
            return Segment::Expand(n);
        }
        // Unparseable -- fall through to drop.
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
            Segment::Drop => {
                // Silently skip, keeping whatever preceded the bracket.
            }
            Segment::Expand(n) => {
                // Repeat the most recent Word/AngleGroup n times total.
                if let Some(item) = output.last().cloned() {
                    for _ in 1..*n {
                        output.push(item.clone());
                    }
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

/// Split a trailing sentence-final punctuation mark off the last word.
///
/// Handles cases like `"cookie."` -> `["cookie", "."]` and `"what?"` ->
/// `["what", "?"]`.
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
// Audible utterance — keeps what was actually spoken
// ---------------------------------------------------------------------------

/// Escape words that are dropped even in the audible view.
///
/// Unidentifiable material (`xxx`, `yyy`, `www` and their suffixed variants) is
/// deliberately absent: it was audible, so it is kept. The two-letter forms
/// `ww`, `xx` and `yy` are here instead: chatter's E241 derives the illegal
/// spellings from that canonical set, so none of the three is a transcribed
/// word to keep.
static AUDIBLE_ESCAPE_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "0", "++", "+<", "+^", "(.)", "(..)", "(...)", ":", ";", ";;", "<", ">", "ww", "xx", "yy",
        "\u{2192}", // →
    ]
    .into_iter()
    .collect()
});

/// Clean section-13 disfluency markers from a single word.
///
/// - Colon between two alphabetic chars: `s:paghetti` → `spaghetti`
/// - Caret: `spa^ghetti` → `spaghetti`
/// - `≠` (U+2260) prefix: `≠butter` → `butter`
/// - Paired `↫` (U+21AB): `like↫ike-ike↫` → `like` (content between
///   matched `↫` pairs is removed; unpaired `↫` is kept as-is)
fn clean_disfluency(word: &str) -> String {
    let mut result = String::with_capacity(word.len());

    // Strip ≠ prefix.
    let word = word.strip_prefix('\u{2260}').unwrap_or(word);

    // Handle paired ↫ markers.
    let word = {
        let first = word.find('\u{21ab}');
        let last = word.rfind('\u{21ab}');
        match (first, last) {
            (Some(f), Some(l)) if f != l => {
                // Paired: keep content before first ↫ and after last ↫.
                let before = &word[..f];
                let after = &word[l + '\u{21ab}'.len_utf8()..];
                std::borrow::Cow::Owned(format!("{before}{after}"))
            }
            _ => std::borrow::Cow::Borrowed(word),
        }
    };

    // Process remaining characters.
    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();
    for (i, &ch) in chars.iter().enumerate() {
        match ch {
            // Caret: always strip.
            '^' => {}
            // Colon: strip only when between two alphabetic characters.
            ':' if i > 0
                && i + 1 < len
                && chars[i - 1].is_alphabetic()
                && chars[i + 1].is_alphabetic() => {}
            _ => result.push(ch),
        }
    }
    result
}

/// Filter and clean individual words for audible output.
///
/// Keeps unidentifiable material (`xxx`, `yyy`, `www`), converts
/// fragments/fillers (`&-uh` -> `uh`), keeps simple events (`&=laughs`) but
/// drops action-only ones (`&=imit:baby`), and removes parenthesized content
/// (the inaudible part) rather than the parentheses alone.
fn filter_words_audible(words: Vec<String>) -> Vec<String> {
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

        // Handle &-prefixed words specially for audible mode.
        if let Some(after_amp) = word.strip_prefix('&') {
            // &=X:Y (action-only simple events like &=imit:baby) — drop.
            if let Some(after_eq) = after_amp.strip_prefix('=') {
                if after_eq.contains(':') {
                    continue; // Non-audible action — drop.
                }
                // &=X (no colon, e.g., &=laughs) — keep as-is.
                let cleaned = clean_disfluency(word);
                if !cleaned.is_empty() {
                    result.push(cleaned);
                }
                continue;
            }
            // &+X, &-X, &~X — strip prefix marker and &.
            // &X (bare) — strip &.
            let rest = after_amp;
            let rest = rest
                .strip_prefix('+')
                .or_else(|| rest.strip_prefix('-'))
                .or_else(|| rest.strip_prefix('~'))
                .unwrap_or(rest);
            if !rest.is_empty() {
                // Strip @ markers and apply disfluency cleaning.
                let rest = match rest.find('@') {
                    Some(pos) => &rest[..pos],
                    None => rest,
                };
                let cleaned = clean_disfluency(rest);
                if !cleaned.is_empty() {
                    result.push(cleaned);
                }
            }
            continue;
        }

        // Filter out escape words, prefixes (except &, handled above), and suffixes.
        let non_amp_escape_prefixes: &[&str] =
            &["[?", "[/", "[<", "[>", "[:", "[!", "[*", "+\"", "+,", "<&"];
        if !AUDIBLE_ESCAPE_WORDS.contains(word)
            && !non_amp_escape_prefixes.iter().any(|e| word.starts_with(e))
            && !ESCAPE_SUFFIXES.iter().any(|e| word.ends_with(e))
        {
            // Strip CHAT special form markers (@b, @c, @o, @s:hu, etc.).
            let word = match word.find('@') {
                Some(pos) => &word[..pos],
                None => word,
            };
            // Remove parenthesized content (the inaudible part).
            // E.g., "(un)til" → "til", "sit(ting)" → "sit".
            let mut cleaned = String::with_capacity(word.len());
            let mut in_parens = false;
            for ch in word.chars() {
                match ch {
                    '(' => in_parens = true,
                    ')' => in_parens = false,
                    _ if !in_parens => cleaned.push(ch),
                    _ => {}
                }
            }
            let cleaned = clean_disfluency(&cleaned);
            if !cleaned.is_empty() {
                result.push(cleaned);
            }
        }
    }
    result
}

/// Produce an audibly faithful transcription of a CHAT utterance.
///
/// This preserves what was actually spoken: repeated/retraced material is kept,
/// unidentifiable material (`xxx`, `yyy`, `www`) is kept, fragments/fillers are
/// included (with prefix markers stripped), and `[x N]` repetitions are
/// expanded.
pub(crate) fn audible_utterance(utterance: &str) -> String {
    let segments = tokenize(utterance);
    let words = process(&segments);
    let mut words = filter_words_audible(words);
    split_trailing_punct(&mut words);
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // -----------------------------------------------------------------------
    // audible_utterance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audible_simple() {
        assert_eq!(audible_utterance("I want cookie ."), "I want cookie .");
    }

    #[test]
    fn test_audible_keeps_xxx() {
        assert_eq!(audible_utterance("xxx ."), "xxx .");
        assert_eq!(audible_utterance("yyy ."), "yyy .");
        assert_eq!(audible_utterance("www ."), "www .");
    }

    #[test]
    fn test_audible_drops_xx() {
        // ww, xx and yy are still escape words in audible mode -- they are the
        // illegal short spellings of www/xxx/yyy, not transcribed words.
        assert_eq!(audible_utterance("ww ."), ".");
        assert_eq!(audible_utterance("xx ."), ".");
        assert_eq!(audible_utterance("yy ."), ".");
    }

    #[test]
    fn test_audible_keeps_repetition() {
        assert_eq!(audible_utterance("the [/] the dog ."), "the the dog .");
    }

    #[test]
    fn test_audible_keeps_repetition_angle_group() {
        assert_eq!(
            audible_utterance("< I wanted > [/] I wanted to invite Margie ."),
            "I wanted I wanted to invite Margie ."
        );
    }

    #[test]
    fn test_audible_keeps_retracing() {
        assert_eq!(
            audible_utterance("< I wanted > [//] blah blah blah ."),
            "I wanted blah blah blah ."
        );
    }

    #[test]
    fn test_audible_keeps_reformulation() {
        assert_eq!(audible_utterance("I [///] she went ."), "I she went .");
    }

    #[test]
    fn test_audible_keeps_false_start() {
        assert_eq!(
            audible_utterance("want [/-] I need cookie ."),
            "want I need cookie ."
        );
    }

    #[test]
    fn test_audible_keeps_excluded() {
        assert_eq!(
            audible_utterance("this is a mor [e] exclude ."),
            "this is a mor exclude ."
        );
    }

    #[test]
    fn test_audible_expansion() {
        assert_eq!(audible_utterance("want [x 3] ."), "want want want .");
    }

    #[test]
    fn test_audible_expansion_single() {
        assert_eq!(audible_utterance("want [x 1] ."), "want .");
    }

    #[test]
    fn test_audible_replacement_keeps_original() {
        // [: replacement] keeps original in audible mode.
        assert_eq!(audible_utterance("goed [: went] ."), "goed .");
    }

    #[test]
    fn test_audible_keep_original_unchanged() {
        // [:: ...] keeps original in both modes.
        assert_eq!(audible_utterance("goed [:: went] ."), "goed .");
    }

    #[test]
    fn test_audible_fragment_prefix_minus() {
        assert_eq!(audible_utterance("&-uh hello ."), "uh hello .");
    }

    #[test]
    fn test_audible_fragment_prefix_plus() {
        assert_eq!(audible_utterance("&+um hello ."), "um hello .");
    }

    #[test]
    fn test_audible_fragment_prefix_tilde() {
        assert_eq!(audible_utterance("&~hey hello ."), "hey hello .");
    }

    #[test]
    fn test_audible_fragment_bare_ampersand() {
        assert_eq!(audible_utterance("&um hello ."), "um hello .");
    }

    #[test]
    fn test_audible_simple_event_kept() {
        assert_eq!(audible_utterance("&=laughs hello ."), "&=laughs hello .");
    }

    #[test]
    fn test_audible_simple_event_action_dropped() {
        // &=imit:baby has the &=X:Y form — non-audible action.
        assert_eq!(audible_utterance("&=imit:baby hello ."), "hello .");
        assert_eq!(audible_utterance("&=ges:ignore hello ."), "hello .");
    }

    #[test]
    fn test_audible_paren_content_removed() {
        // Audible removes parenthesized content (the inaudible part).
        assert_eq!(audible_utterance("(un)til the end ."), "til the end .");
        assert_eq!(audible_utterance("sit(ting) down ."), "sit down .");
        assert_eq!(audible_utterance("(be)cause ."), "cause .");
    }

    #[test]
    fn test_audible_disfluency_colon() {
        assert_eq!(audible_utterance("s:paghetti ."), "spaghetti .");
    }

    #[test]
    fn test_audible_disfluency_caret() {
        assert_eq!(audible_utterance("spa^ghetti ."), "spaghetti .");
    }

    #[test]
    fn test_audible_disfluency_not_equal() {
        assert_eq!(audible_utterance("\u{2260}butter ."), "butter .");
    }

    #[test]
    fn test_audible_disfluency_leftwards_arrow_paired() {
        assert_eq!(audible_utterance("like\u{21ab}ike-ike\u{21ab} ."), "like .");
    }

    #[test]
    fn test_audible_disfluency_leftwards_arrow_unpaired() {
        // Unpaired ↫ is kept as-is.
        assert_eq!(audible_utterance("like\u{21ab} ."), "like\u{21ab} .");
    }

    #[test]
    fn test_audible_omitted_words_still_filtered() {
        assert_eq!(audible_utterance("0you go ."), "go .");
    }

    #[test]
    fn test_audible_at_markers_stripped() {
        assert_eq!(audible_utterance("bingbing@c ."), "bingbing .");
    }

    #[test]
    fn test_audible_drops_annotations() {
        assert_eq!(
            audible_utterance("I want [= desire] cookie ."),
            "I want cookie ."
        );
    }

    #[test]
    fn test_audible_drops_timestamps() {
        let input = "hello \x15123_456\x15 .";
        assert_eq!(audible_utterance(input), "hello .");
    }

    #[test]
    fn test_audible_drops_pauses() {
        assert_eq!(audible_utterance("hello (1.5) world ."), "hello world .");
        assert_eq!(audible_utterance("hello (.) world ."), "hello world .");
    }

    // -----------------------------------------------------------------------
    // clean_disfluency unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clean_disfluency_colon_between_alpha() {
        assert_eq!(clean_disfluency("s:paghetti"), "spaghetti");
    }

    #[test]
    fn test_clean_disfluency_colon_not_between_alpha() {
        // Colon at end (like xxx:) should not be stripped.
        assert_eq!(clean_disfluency("xxx:"), "xxx:");
    }

    #[test]
    fn test_clean_disfluency_caret() {
        assert_eq!(clean_disfluency("spa^ghetti"), "spaghetti");
    }

    #[test]
    fn test_clean_disfluency_not_equal_prefix() {
        assert_eq!(clean_disfluency("\u{2260}butter"), "butter");
    }

    #[test]
    fn test_clean_disfluency_paired_arrow() {
        assert_eq!(clean_disfluency("like\u{21ab}ike-ike\u{21ab}"), "like");
    }

    #[test]
    fn test_clean_disfluency_unpaired_arrow() {
        assert_eq!(clean_disfluency("like\u{21ab}"), "like\u{21ab}");
    }

    #[test]
    fn test_clean_disfluency_no_change() {
        assert_eq!(clean_disfluency("hello"), "hello");
    }
}
