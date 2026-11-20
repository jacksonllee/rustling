//! Build a [`ChatFile`] from CHAT text that rustling itself generated.
//!
//! The SRT, ELAN and TextGrid converters emit CHAT whose utterance bodies are
//! arbitrary source text -- a subtitle line, an annotation value -- rather than
//! CHAT markup. Reading that back with a CHAT grammar treats punctuation as
//! structure and silently drops whatever it cannot account for: `Hello [world]
//! and <stuff> & more!` comes back as `more !`, `50% off` as `off`, and `cost
//! $5` as `cost`, because `[`, `<`, `&`, `%` and `$` all open CHAT constructs.
//!
//! These files carry no morphology and no annotation to recover, so the words
//! are simply the whitespace-separated runs of the source text. That is the
//! same reasoning that put the CoNLL-U converter on
//! [`conllu_file_to_chat_file`](crate::conllu::chat_writer), which builds its
//! `ChatFile` from the token data instead of re-parsing its own output.

use std::collections::HashMap;

use crate::chat::header::{Headers, Participant};
use crate::chat::reader::ChatFile;
use crate::chat::utterance::{Token, Utterance};

/// Build a [`ChatFile`] from CHAT text one of rustling's own converters wrote.
///
/// `text` is expected in the shape those converters emit: `@UTF8` / `@Begin`,
/// an optional `@Participants` line, then `*CODE:\t…` main tiers each
/// optionally followed by `%tier:\t…` lines, and `@End`. Lines are kept
/// verbatim as `raw_lines`, so the serialized form and the in-memory events
/// cannot drift apart.
pub(crate) fn chat_file_from_generated(text: &str, id: String) -> ChatFile {
    let mut headers = Headers::default();
    let mut events: Vec<Utterance> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();

    for line in text.lines() {
        raw_lines.push(line.to_string());

        if let Some(rest) = line.strip_prefix("@Participants:") {
            headers.participants = parse_participants(rest);
        } else if let Some(rest) = line.strip_prefix('*') {
            let Some((code, body)) = rest.split_once(':') else {
                continue;
            };
            let code = code.trim().to_string();
            let body = body.trim();
            let (spoken, time_marks) = split_time_mark(body);

            let mut tiers = HashMap::new();
            // The tier text keeps the media bullet, matching what a parsed
            // transcript exposes through `Utterance.tiers`.
            tiers.insert(code.clone(), body.to_string());

            events.push(Utterance {
                participant: Some(code),
                tokens: Some(tokenize(spoken)),
                time_marks,
                tiers: Some(tiers),
                changeable_header: None,
                mor_tier_name: None,
                gra_tier_name: None,
            });
        } else if line.starts_with('%')
            && let Some((name, body)) = line.split_once(':')
            && let Some(utt) = events.last_mut()
            && let Some(tiers) = utt.tiers.as_mut()
        {
            tiers.insert(name.trim().to_string(), body.trim().to_string());
        }
    }

    ChatFile::new(id, headers, events, raw_lines)
}

/// Parse an `@Participants` body: `CHI Child, MOT Mary Mother`.
///
/// CHAT writes an entry as `CODE Role` or `CODE Name Role`, so a lone trailing
/// field is the role and the name is only present once there are two. Reading
/// it the way chatter does is what keeps `to_chat` agreeing with
/// `to_chat_files`, whose output is parsed by chatter on the way back in: an
/// ELAN `PARTICIPANT` of `Mary Smith` yields `CHI Mary Smith`, which is a
/// name and a role rather than a two-word role.
fn parse_participants(rest: &str) -> Vec<Participant> {
    rest.split(',')
        .filter_map(|entry| {
            let mut parts = entry.split_whitespace();
            let code = parts.next()?;
            let mut fields = parts.collect::<Vec<_>>();
            let name = if fields.len() > 1 {
                fields.remove(0).to_string()
            } else {
                String::new()
            };
            Some(Participant {
                code: code.to_string(),
                name,
                role: fields.join(" "),
                ..Participant::default()
            })
        })
        .collect()
}

/// Split a trailing `\x15start_end\x15` media bullet off a main tier body.
fn split_time_mark(body: &str) -> (&str, Option<(i64, i64)>) {
    const BULLET: char = '\u{15}';
    let Some(open) = body.rfind(BULLET) else {
        return (body, None);
    };
    // The bullet is the last thing on the line, so the opening marker is the
    // second-to-last `\x15` and the closing one ends the body.
    let Some(open) = body[..open].rfind(BULLET) else {
        return (body, None);
    };
    let inner = body[open + BULLET.len_utf8()..].trim_end_matches(BULLET);
    let Some((start, end)) = inner.split_once('_') else {
        return (body, None);
    };
    match (start.trim().parse(), end.trim().parse()) {
        (Ok(start), Ok(end)) => (body[..open].trim_end(), Some((start, end))),
        _ => (body, None),
    }
}

/// Split source text into word tokens.
///
/// Whitespace runs are the word boundaries, so nothing in the source is
/// dropped. The one concession to CHAT is the terminator: a trailing run of
/// `.`, `?` or `!` on the final word becomes its own token, which is where a
/// parsed transcript puts it and what `Hello world.` has always produced.
fn tokenize(text: &str) -> Vec<Token> {
    let mut words: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    if let Some(last) = words.last_mut() {
        let stem = last.trim_end_matches(['.', '?', '!']);
        if !stem.is_empty() && stem.len() < last.len() {
            let terminator = last[stem.len()..].to_string();
            last.truncate(stem.len());
            words.push(terminator);
        }
    }
    words
        .into_iter()
        .map(|word| Token {
            word,
            pos: None,
            mor: None,
            gra: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn words(file: &ChatFile) -> Vec<&str> {
        file.real_utterances()
            .flat_map(|u| u.tokens.as_ref().unwrap())
            .map(|t| t.word.as_str())
            .collect()
    }

    fn build(body: &str) -> ChatFile {
        let text = format!("@UTF8\n@Begin\n@Participants:\tSPK Speaker\n*SPK:\t{body}\n@End\n");
        chat_file_from_generated(&text, "x.cha".to_string())
    }

    /// Every one of these loses words when read back through a CHAT grammar,
    /// which is the whole reason this module exists.
    #[test]
    fn chat_metacharacters_survive() {
        assert_eq!(
            words(&build("Hello [world] and <stuff> & more!")),
            ["Hello", "[world]", "and", "<stuff>", "&", "more", "!"]
        );
        assert_eq!(words(&build("50% off")), ["50%", "off"]);
        assert_eq!(words(&build("cost $5")), ["cost", "$5"]);
        assert_eq!(words(&build("x = 3")), ["x", "=", "3"]);
        assert_eq!(words(&build("He said \"hi\"")), ["He", "said", "\"hi\""]);
        assert_eq!(words(&build("Well... maybe")), ["Well...", "maybe"]);
    }

    #[test]
    fn terminator_becomes_its_own_token() {
        assert_eq!(words(&build("Hello world.")), ["Hello", "world", "."]);
        assert_eq!(words(&build("Really?")), ["Really", "?"]);
        assert_eq!(words(&build("Stop!")), ["Stop", "!"]);
        // A word that is nothing but terminator characters stays whole.
        assert_eq!(words(&build("wait ...")), ["wait", "..."]);
    }

    #[test]
    fn time_mark_is_read_and_kept_in_tier_text() {
        let file = build("Hello world. \u{15}0_1500\u{15}");
        let utt = file.real_utterances().next().unwrap();
        assert_eq!(utt.time_marks, Some((0, 1500)));
        assert_eq!(
            utt.tiers.as_ref().unwrap().get("SPK").map(String::as_str),
            Some("Hello world. \u{15}0_1500\u{15}")
        );
        assert_eq!(words(&file), ["Hello", "world", "."]);
    }

    #[test]
    fn participants_and_dependent_tiers() {
        let text = "@UTF8\n@Begin\n@Participants:\tCHI Target_Child, MOT Mother\n\
            *CHI:\thi .\n%com:\tsmiling\n*MOT:\tbye .\n@End\n";
        let file = chat_file_from_generated(text, "x.cha".to_string());
        let codes: Vec<&str> = file
            .headers
            .participants
            .iter()
            .map(|p| p.code.as_str())
            .collect();
        assert_eq!(codes, ["CHI", "MOT"]);
        assert_eq!(file.headers.participants[0].role, "Target_Child");
        let utts: Vec<_> = file.real_utterances().collect();
        assert_eq!(utts.len(), 2);
        assert_eq!(
            utts[0]
                .tiers
                .as_ref()
                .unwrap()
                .get("%com")
                .map(String::as_str),
            Some("smiling")
        );
        assert!(!utts[1].tiers.as_ref().unwrap().contains_key("%com"));
        assert_eq!(file.raw_lines.len(), 7);
    }

    #[test]
    fn multi_word_participant_splits_into_name_and_role() {
        // An ELAN `PARTICIPANT` of `Mary Smith` reaches CHAT as a third field,
        // which is a name followed by a role -- the reading chatter gives it
        // when `to_chat_files` output is parsed back in.
        let text = "@UTF8\n@Begin\n@Participants:\tCHI Mary Smith, MOT Mother\n\
            *CHI:\thi .\n@End\n";
        let file = chat_file_from_generated(text, "x.cha".to_string());
        let chi = &file.headers.participants[0];
        assert_eq!((chi.name.as_str(), chi.role.as_str()), ("Mary", "Smith"));
        let mot = &file.headers.participants[1];
        assert_eq!((mot.name.as_str(), mot.role.as_str()), ("", "Mother"));
    }
}
