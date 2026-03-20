//! Convert SRT data to CHAT format.

use super::reader::SrtFile;

/// Convert a single [`SrtFile`] to a CHAT format string.
///
/// Uses a default participant code `"SPK"` since SRT has no participant info.
pub(crate) fn srt_file_to_chat_str(file: &SrtFile) -> String {
    let mut output = String::with_capacity(4096);
    output.push_str("@UTF8\n");
    output.push_str("@Begin\n");
    output.push_str("@Participants:\tSPK Speaker\n");

    for block in &file.blocks {
        // Join multiline subtitle text with space for CHAT (single-line utterances).
        let text = block.text.replace('\n', " ");
        output.push_str(&format!(
            "*SPK:\t{text} \x15{start}_{end}\x15\n",
            start = block.start_ms,
            end = block.end_ms,
        ));
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
    use crate::srt::reader::{SrtBlock, SrtFile};

    fn make_srt_file(blocks: Vec<SrtBlock>) -> SrtFile {
        SrtFile {
            file_path: "test.srt".to_string(),
            blocks,
        }
    }

    #[test]
    fn test_basic_conversion() {
        let file = make_srt_file(vec![SrtBlock {
            index: 1,
            text: "Hello world.".to_string(),
            start_ms: 0,
            end_ms: 1500,
        }]);
        let chat = srt_file_to_chat_str(&file);

        assert!(chat.contains("@UTF8"));
        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@Participants:\tSPK Speaker"));
        assert!(chat.contains("*SPK:\tHello world. \x150_1500\x15"));
        assert!(chat.contains("@End"));
    }

    #[test]
    fn test_multiline_text_joined() {
        let file = make_srt_file(vec![SrtBlock {
            index: 1,
            text: "Senator, we're making\nour final approach.".to_string(),
            start_ms: 136612,
            end_ms: 139376,
        }]);
        let chat = srt_file_to_chat_str(&file);

        // Multiline text should be joined with space.
        assert!(
            chat.contains("*SPK:\tSenator, we're making our final approach. \x15136612_139376\x15")
        );
    }

    #[test]
    fn test_multiple_blocks() {
        let file = make_srt_file(vec![
            SrtBlock {
                index: 1,
                text: "First line.".to_string(),
                start_ms: 0,
                end_ms: 1000,
            },
            SrtBlock {
                index: 2,
                text: "Second line.".to_string(),
                start_ms: 1500,
                end_ms: 3000,
            },
        ]);
        let chat = srt_file_to_chat_str(&file);

        assert!(chat.contains("*SPK:\tFirst line. \x150_1000\x15"));
        assert!(chat.contains("*SPK:\tSecond line. \x151500_3000\x15"));
    }

    #[test]
    fn test_empty_file() {
        let file = make_srt_file(vec![]);
        let chat = srt_file_to_chat_str(&file);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
        assert!(chat.contains("@Participants:\tSPK Speaker"));
    }

    #[test]
    fn test_round_trip_via_chat_parser() {
        let file = make_srt_file(vec![
            SrtBlock {
                index: 1,
                text: "Hello world.".to_string(),
                start_ms: 0,
                end_ms: 1500,
            },
            SrtBlock {
                index: 2,
                text: "Goodbye.".to_string(),
                start_ms: 2000,
                end_ms: 3500,
            },
        ]);
        let chat_str = srt_file_to_chat_str(&file);

        // Parse the generated CHAT string.
        let (chat, _) = crate::chat::Chat::from_strs(vec![chat_str], None, false, None, None);
        let files = chat.files();
        assert_eq!(files.len(), 1);

        let utts: Vec<_> = files[0].real_utterances().collect();
        assert_eq!(utts.len(), 2);
        assert_eq!(utts[0].participant.as_deref(), Some("SPK"));
        assert_eq!(utts[0].time_marks, Some((0, 1500)));
        assert_eq!(utts[1].participant.as_deref(), Some("SPK"));
        assert_eq!(utts[1].time_marks, Some((2000, 3500)));
    }
}
