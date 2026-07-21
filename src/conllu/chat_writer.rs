//! Convert CoNLL-U data to CHAT format.

use std::collections::HashMap;

use super::reader::{ConlluFile, Sentence};
use crate::chat::{ChatFile, Headers, Participant, Token, Utterance};

/// The surface words, `%mor` text, and `%gra` text of one CoNLL-U sentence.
///
/// The surface text uses multiword-token forms (skipping their component
/// words); `%mor`/`%gra` are built from the non-multiword, non-empty-node
/// component tokens (`UPOS|LEMMA[&FEATS]` and `ID|HEAD|DEPREL`).
fn sentence_parts(sentence: &Sentence) -> (Vec<String>, String, String) {
    let mut words: Vec<String> = Vec::new();
    let mut skip_until: Option<usize> = None;
    for token in &sentence.tokens {
        if token.is_empty_node() {
            continue;
        }
        if let Some(end) = skip_until
            && let Ok(id_num) = token.id.parse::<usize>()
        {
            if id_num <= end {
                continue;
            }
            skip_until = None;
        }
        if token.is_multiword()
            && let Some(dash_pos) = token.id.find('-')
            && let Ok(end) = token.id[dash_pos + 1..].parse::<usize>()
        {
            skip_until = Some(end);
        }
        words.push(token.form.clone());
    }

    let mut mor_parts: Vec<String> = Vec::new();
    let mut gra_parts: Vec<String> = Vec::new();
    for token in &sentence.tokens {
        if token.is_multiword() || token.is_empty_node() {
            continue;
        }
        let mor = if token.feats != "_" && !token.feats.is_empty() {
            format!("{}|{}&{}", token.upos, token.lemma, token.feats)
        } else {
            format!("{}|{}", token.upos, token.lemma)
        };
        mor_parts.push(mor);
        gra_parts.push(format!("{}|{}|{}", token.id, token.head, token.deprel));
    }

    (words, mor_parts.join(" "), gra_parts.join(" "))
}

/// Convert a single [`ConlluFile`] to a CHAT format string.
///
/// Uses a default participant code `"SPK"` since CoNLL-U has no participant info.
/// Morphology and grammatical relations are generated from CoNLL-U token fields:
/// - `%mor`: `UPOS|LEMMA` (with `&FEATS` appended if features are present)
/// - `%gra`: `ID|HEAD|DEPREL`
///
/// Multiword tokens (range IDs like "1-2") and empty nodes (decimal IDs like "1.1")
/// are skipped in `%mor` and `%gra` tiers since they carry no analysis.
pub(crate) fn conllu_file_to_chat_str(file: &ConlluFile) -> String {
    let mut output = String::with_capacity(4096);
    output.push_str("@UTF8\n");
    output.push_str("@Begin\n");
    output.push_str("@Participants:\tSPK Speaker\n");

    for sentence in &file.sentences {
        let (words, mor, gra) = sentence_parts(sentence);
        output.push_str(&format!("*SPK:\t{}\n", words.join(" ")));
        if !mor.is_empty() {
            output.push_str(&format!("%mor:\t{mor}\n"));
        }
        if !gra.is_empty() {
            output.push_str(&format!("%gra:\t{gra}\n"));
        }
    }

    output.push_str("@End\n");
    output
}

/// Convert a single [`ConlluFile`] directly into a rustling [`ChatFile`].
///
/// The events are built from the CoNLL-U token data directly (bypassing a
/// CHAT-text round-trip through the parser). This avoids re-parsing arbitrary
/// treebank surface text — which may contain CHAT-special characters (`[`, `<`,
/// `&`, …) that the parser cannot represent — while still producing the same
/// word-only utterances the string round-trip did (mor/gra are kept as raw tier
/// text, matching the `mor_tier=None` behavior of the CHAT conversion path).
pub(crate) fn conllu_file_to_chat_file(file: &ConlluFile, id: String) -> ChatFile {
    let raw = conllu_file_to_chat_str(file);
    let raw_lines: Vec<String> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();

    let headers = Headers {
        participants: vec![Participant {
            code: "SPK".to_string(),
            role: "Speaker".to_string(),
            ..Participant::default()
        }],
        ..Headers::default()
    };

    let mut events = Vec::with_capacity(file.sentences.len());
    for sentence in &file.sentences {
        let (words, mor, gra) = sentence_parts(sentence);
        let tokens: Vec<Token> = words
            .iter()
            .map(|w| Token {
                word: w.clone(),
                pos: None,
                mor: None,
                gra: None,
            })
            .collect();
        let mut tiers = HashMap::new();
        tiers.insert("SPK".to_string(), words.join(" "));
        if !mor.is_empty() {
            tiers.insert("%mor".to_string(), mor);
        }
        if !gra.is_empty() {
            tiers.insert("%gra".to_string(), gra);
        }
        events.push(Utterance {
            participant: Some("SPK".to_string()),
            tokens: Some(tokens),
            time_marks: None,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: None,
            gra_tier_name: None,
        });
    }

    ChatFile::new(id, headers, events, raw_lines)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::BaseChat;
    use crate::conllu::reader::{ConlluFile, ConlluToken, Sentence};

    fn make_token(
        id: &str,
        form: &str,
        lemma: &str,
        upos: &str,
        feats: &str,
        head: &str,
        deprel: &str,
    ) -> ConlluToken {
        ConlluToken {
            id: id.to_string(),
            form: form.to_string(),
            lemma: lemma.to_string(),
            upos: upos.to_string(),
            xpos: "_".to_string(),
            feats: feats.to_string(),
            head: head.to_string(),
            deprel: deprel.to_string(),
            deps: "_".to_string(),
            misc: "_".to_string(),
        }
    }

    #[test]
    fn test_basic_conversion() {
        let file = ConlluFile {
            file_path: "test.conllu".to_string(),
            sentences: vec![Sentence {
                comments: Some(vec!["sent_id = 1".to_string()]),
                tokens: vec![
                    make_token("1", "The", "the", "DET", "Definite=Def", "2", "det"),
                    make_token("2", "cat", "cat", "NOUN", "Number=Sing", "3", "nsubj"),
                    make_token(
                        "3",
                        "sat",
                        "sit",
                        "VERB",
                        "Mood=Ind|Tense=Past",
                        "0",
                        "root",
                    ),
                    make_token("4", ".", ".", "PUNCT", "_", "3", "punct"),
                ],
            }],
        };
        let chat = conllu_file_to_chat_str(&file);

        assert!(chat.contains("@UTF8"));
        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@Participants:\tSPK Speaker"));
        assert!(chat.contains("*SPK:\tThe cat sat ."));
        assert!(chat.contains(
            "%mor:\tDET|the&Definite=Def NOUN|cat&Number=Sing VERB|sit&Mood=Ind|Tense=Past PUNCT|."
        ));
        assert!(chat.contains("%gra:\t1|2|det 2|3|nsubj 3|0|root 4|3|punct"));
        assert!(chat.contains("@End"));
    }

    #[test]
    fn test_multiword_skipped_in_mor_gra() {
        let file = ConlluFile {
            file_path: "test.conllu".to_string(),
            sentences: vec![Sentence {
                comments: None,
                tokens: vec![
                    make_token("1", "Go", "ir", "VERB", "_", "0", "root"),
                    ConlluToken {
                        id: "2-3".to_string(),
                        form: "al".to_string(),
                        lemma: "_".to_string(),
                        upos: "_".to_string(),
                        xpos: "_".to_string(),
                        feats: "_".to_string(),
                        head: "_".to_string(),
                        deprel: "_".to_string(),
                        deps: "_".to_string(),
                        misc: "_".to_string(),
                    },
                    make_token("2", "a", "a", "ADP", "_", "4", "case"),
                    make_token("3", "el", "el", "DET", "_", "4", "det"),
                    make_token("4", "mar", "mar", "NOUN", "_", "1", "obl"),
                ],
            }],
        };
        let chat = conllu_file_to_chat_str(&file);

        // Surface text uses "al" (multiword form), not "a el".
        assert!(chat.contains("*SPK:\tGo al mar"));
        // %mor should not include the multiword token.
        let mor_line = chat.lines().find(|l| l.starts_with("%mor:")).unwrap();
        assert!(!mor_line.contains("_|_"));
        assert!(mor_line.contains("VERB|ir"));
        assert!(mor_line.contains("ADP|a"));
    }

    #[test]
    fn test_empty_file() {
        let file = ConlluFile {
            file_path: "empty.conllu".to_string(),
            sentences: vec![],
        };
        let chat = conllu_file_to_chat_str(&file);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
    }

    #[test]
    fn test_round_trip_via_chat_parser() {
        let file = ConlluFile {
            file_path: "test.conllu".to_string(),
            sentences: vec![Sentence {
                comments: None,
                tokens: vec![
                    make_token("1", "Hello", "hello", "NOUN", "_", "0", "root"),
                    make_token("2", "world", "world", "NOUN", "_", "1", "flat"),
                    make_token("3", ".", ".", "PUNCT", "_", "1", "punct"),
                ],
            }],
        };
        let chat_str = conllu_file_to_chat_str(&file);

        let (chat, _) =
            crate::chat::Chat::from_strs(vec![chat_str], None, false, None, None, false);
        let files = chat.files();
        assert_eq!(files.len(), 1);
        let utts: Vec<_> = files[0].real_utterances().collect();
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].participant.as_deref(), Some("SPK"));
    }
}
