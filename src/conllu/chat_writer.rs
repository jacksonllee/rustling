//! Convert CoNLL-U data to CHAT format.

use super::reader::ConlluFile;

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
        // Build utterance text from FORM fields.
        // Use multiword tokens for surface form when available, skip their
        // component words in the surface text.
        let mut text_parts: Vec<&str> = Vec::new();
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
            if token.is_multiword() {
                // Parse range end (e.g., "2-3" -> 3).
                if let Some(dash_pos) = token.id.find('-')
                    && let Ok(end) = token.id[dash_pos + 1..].parse::<usize>()
                {
                    skip_until = Some(end);
                }
            }
            text_parts.push(&token.form);
        }
        let text = text_parts.join(" ");
        output.push_str(&format!("*SPK:\t{text}\n"));

        // Build %mor tier from non-multiword, non-empty-node tokens.
        let mut mor_parts: Vec<String> = Vec::new();
        let mut gra_parts: Vec<String> = Vec::new();
        for token in &sentence.tokens {
            if token.is_multiword() || token.is_empty_node() {
                continue;
            }
            // %mor: UPOS|LEMMA or UPOS|LEMMA&FEATS
            let mor = if token.feats != "_" && !token.feats.is_empty() {
                format!("{}|{}&{}", token.upos, token.lemma, token.feats)
            } else {
                format!("{}|{}", token.upos, token.lemma)
            };
            mor_parts.push(mor);

            // %gra: ID|HEAD|DEPREL
            let gra = format!("{}|{}|{}", token.id, token.head, token.deprel);
            gra_parts.push(gra);
        }

        if !mor_parts.is_empty() {
            output.push_str(&format!("%mor:\t{}\n", mor_parts.join(" ")));
        }
        if !gra_parts.is_empty() {
            output.push_str(&format!("%gra:\t{}\n", gra_parts.join(" ")));
        }
    }

    output.push_str("@End\n");
    output
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

        let (chat, _) = crate::chat::Chat::from_strs(vec![chat_str], None, false, None, None);
        let files = chat.files();
        assert_eq!(files.len(), 1);
        let utts: Vec<_> = files[0].real_utterances().collect();
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].participant.as_deref(), Some("SPK"));
    }
}
