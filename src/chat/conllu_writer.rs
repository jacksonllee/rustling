//! Convert CHAT data to CoNLL-U format.

use super::reader::ChatFile;

/// Convert a single [`ChatFile`] to a CoNLL-U format string.
///
/// Each real utterance (non-changeable-header) becomes one sentence.
/// Token fields are mapped as:
/// - FORM ← `Token.word`
/// - UPOS ← `Token.pos` (or `_` if absent)
/// - LEMMA ← `Token.mor` (or `_` if absent)
/// - HEAD ← `Token.gra.head` (or `_`)
/// - DEPREL ← `Token.gra.rel` (or `_`)
/// - ID ← sequential 1-based index (or `Token.gra.dep` when available)
///
/// Fields without a direct mapping (XPOS, FEATS, DEPS, MISC) are set to `_`.
pub(crate) fn chat_file_to_conllu_str(file: &ChatFile) -> String {
    let mut output = String::with_capacity(4096);
    let mut sent_id = 0usize;

    for utt in file.real_utterances() {
        sent_id += 1;

        // Reconstruct the utterance text from tokens if available.
        let text = utt
            .tokens
            .as_ref()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter(|t| !t.word.is_empty())
                    .map(|t| t.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();

        if !output.is_empty() {
            output.push('\n');
        }

        output.push_str(&format!("# sent_id = {sent_id}\n"));
        if !text.is_empty() {
            output.push_str(&format!("# text = {text}\n"));
        }

        if let Some(tokens) = &utt.tokens {
            for (word_idx, token) in tokens.iter().enumerate() {
                let word_id = word_idx + 1;

                // Use gra.dep as the ID if available, otherwise sequential.
                let id = token
                    .gra
                    .as_ref()
                    .map_or_else(|| word_id.to_string(), |g| g.dep.to_string());

                let form = if token.word.is_empty() {
                    "_"
                } else {
                    &token.word
                };
                let upos = token.pos.as_deref().unwrap_or("_");
                let lemma = token.mor.as_deref().unwrap_or("_");
                let head = token
                    .gra
                    .as_ref()
                    .map_or("_".to_string(), |g| g.head.to_string());
                let deprel = token.gra.as_ref().map_or("_", |g| g.rel.as_str());

                output.push_str(&format!(
                    "{id}\t{form}\t{lemma}\t{upos}\t_\t_\t{head}\t{deprel}\t_\t_\n"
                ));
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{BaseChat, Chat};

    #[test]
    fn test_basic_conversion() {
        // Use CHAT's native %mor format: lowercase POS tags, punctuation is bare.
        let chat_str = "@UTF8\n@Begin\n@Participants:\tCHI Child\n*CHI:\tI want cookie .\n%mor:\tpro|I v|want n|cookie .\n%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n@End\n";
        let (chat, _) = Chat::from_strs(
            vec![chat_str.to_string()],
            None,
            false,
            Some("%mor"),
            Some("%gra"),
        );
        let conllu = chat_file_to_conllu_str(&chat.files()[0]);

        assert!(conllu.contains("# sent_id = 1"));
        assert!(conllu.contains("# text = I want cookie ."));
        // Check token lines: id, form, lemma, upos, _, _, head, deprel, _, _
        assert!(conllu.contains("1\tI\tI\tpro\t_\t_\t2\tSUBJ\t_\t_"));
        assert!(conllu.contains("2\twant\twant\tv\t_\t_\t0\tROOT\t_\t_"));
        assert!(conllu.contains("3\tcookie\tcookie\tn\t_\t_\t2\tOBJ\t_\t_"));
        // CHAT punctuation has empty pos, so upos field is empty string.
        assert!(conllu.contains("4\t.\t.\t\t_\t_\t2\tPUNCT\t_\t_"));
    }

    #[test]
    fn test_multiple_utterances() {
        let chat_str = "@UTF8\n@Begin\n@Participants:\tCHI Child, MOT Mother\n*CHI:\tI want cookie .\n%mor:\tpro|I v|want n|cookie .\n%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n*MOT:\tno .\n%mor:\tco|no .\n%gra:\t1|0|ROOT 2|1|PUNCT\n@End\n";
        let (chat, _) = Chat::from_strs(
            vec![chat_str.to_string()],
            None,
            false,
            Some("%mor"),
            Some("%gra"),
        );
        let conllu = chat_file_to_conllu_str(&chat.files()[0]);

        assert!(conllu.contains("# sent_id = 1"));
        assert!(conllu.contains("# sent_id = 2"));
    }

    #[test]
    fn test_no_mor_gra() {
        let chat_str = "@UTF8\n@Begin\n@Participants:\tSPK Speaker\n*SPK:\tHello world .\n@End\n";
        let (chat, _) = Chat::from_strs(vec![chat_str.to_string()], None, false, None, None);
        let conllu = chat_file_to_conllu_str(&chat.files()[0]);

        assert!(conllu.contains("# sent_id = 1"));
        // Tokens should have _ for upos, lemma, head, deprel.
        assert!(conllu.contains("1\tHello\t_\t_\t_\t_\t_\t_\t_\t_"));
    }

    #[test]
    fn test_empty_file() {
        let chat_str = "@UTF8\n@Begin\n@Participants:\tSPK Speaker\n@End\n";
        let (chat, _) = Chat::from_strs(vec![chat_str.to_string()], None, false, None, None);
        let conllu = chat_file_to_conllu_str(&chat.files()[0]);

        assert!(conllu.is_empty() || !conllu.contains("# sent_id"));
    }
}
