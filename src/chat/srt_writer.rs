//! Convert CHAT data to SRT (SubRip Subtitle) format.

use super::reader::ChatFile;
use crate::srt::format_srt_time;

/// A single subtitle entry collected from CHAT utterances.
struct SrtEntry {
    text: String,
    start_ms: i64,
    end_ms: i64,
}

/// Convert a single [`ChatFile`] to an SRT format string.
///
/// If `participants` is `Some`, only those participant codes are included.
/// If `None`, all participants are included.
/// Utterances without time marks are skipped (SRT requires time ranges).
/// When multiple participants are present, the subtitle text is prefixed
/// with the participant code (e.g., `"CHI: more cookie ."`).
pub(crate) fn chat_file_to_srt_str(file: &ChatFile, participants: Option<&[String]>) -> String {
    // Collect participant codes that appear.
    let mut seen_participants: Vec<String> = Vec::new();

    // Collect entries.
    let mut entries: Vec<SrtEntry> = Vec::new();

    for utt in file.real_utterances() {
        let participant = match &utt.participant {
            Some(p) => p,
            None => continue,
        };

        // Filter by requested participants.
        if let Some(codes) = participants
            && !codes.iter().any(|c| c == participant)
        {
            continue;
        }

        // Skip utterances without time marks.
        let (start, end) = match utt.time_marks {
            Some((s, e)) => (s, e),
            None => continue,
        };

        if !seen_participants.contains(participant) {
            seen_participants.push(participant.clone());
        }

        // Get the main text from tiers.
        let text = utt
            .tiers
            .as_ref()
            .and_then(|t| t.get(participant))
            .cloned()
            .unwrap_or_default();

        // Strip CHAT bullet markers from text: \x15start_end\x15
        let text = strip_bullet_markers(&text);

        entries.push(SrtEntry {
            text: format_subtitle_text(&text, participant, seen_participants.len() > 1),
            start_ms: start,
            end_ms: end,
        });
    }

    // Sort by start time, then end time.
    entries.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    // Check if we need participant prefixes (re-evaluate after filtering).
    let multi_participant = seen_participants.len() > 1;
    // If multi_participant changed after collecting, rebuild text. But we already
    // formatted above with the knowledge of > 1 participant; the entries are already correct
    // because we push to seen_participants before creating entries for subsequent participants,
    // BUT the first participant's entries won't have the prefix if they were added
    // before seeing a second participant. Re-format if needed.
    if multi_participant {
        // Already correctly formatted since we check `seen_participants.len() > 1`
        // at entry creation time. However, the first entries for the first participant
        // were created when `seen_participants.len() == 1`. Let's just regenerate.
        return generate_srt_from_chat_file(file, participants);
    }

    // Generate SRT output.
    let mut output = String::with_capacity(4096);
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format!("{}\n", i + 1));
        output.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(entry.start_ms),
            format_srt_time(entry.end_ms),
        ));
        output.push_str(&entry.text);
        output.push('\n');
    }
    output
}

/// Full generation that knows participant count upfront.
fn generate_srt_from_chat_file(file: &ChatFile, participants: Option<&[String]>) -> String {
    // First pass: determine which participants are present.
    let mut seen_participants: Vec<String> = Vec::new();
    for utt in file.real_utterances() {
        if let Some(participant) = &utt.participant {
            if let Some(codes) = participants
                && !codes.iter().any(|c| c == participant)
            {
                continue;
            }
            if utt.time_marks.is_some() && !seen_participants.contains(participant) {
                seen_participants.push(participant.clone());
            }
        }
    }

    let multi = seen_participants.len() > 1;

    // Second pass: collect entries.
    let mut entries: Vec<SrtEntry> = Vec::new();
    for utt in file.real_utterances() {
        let participant = match &utt.participant {
            Some(p) => p,
            None => continue,
        };
        if let Some(codes) = participants
            && !codes.iter().any(|c| c == participant)
        {
            continue;
        }
        let (start, end) = match utt.time_marks {
            Some((s, e)) => (s, e),
            None => continue,
        };

        let text = utt
            .tiers
            .as_ref()
            .and_then(|t| t.get(participant))
            .cloned()
            .unwrap_or_default();
        let text = strip_bullet_markers(&text);

        entries.push(SrtEntry {
            text: format_subtitle_text(&text, participant, multi),
            start_ms: start,
            end_ms: end,
        });
    }

    entries.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    let mut output = String::with_capacity(4096);
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format!("{}\n", i + 1));
        output.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(entry.start_ms),
            format_srt_time(entry.end_ms),
        ));
        output.push_str(&entry.text);
        output.push('\n');
    }
    output
}

/// Format subtitle text, optionally prefixing with participant code.
fn format_subtitle_text(text: &str, participant: &str, multi_participant: bool) -> String {
    if multi_participant {
        format!("{participant}: {text}")
    } else {
        text.to_string()
    }
}

/// Strip CHAT bullet markers (`\x15start_end\x15`) from text.
fn strip_bullet_markers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_bullet = false;
    for ch in text.chars() {
        if ch == '\x15' {
            in_bullet = !in_bullet;
        } else if !in_bullet {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::header::Headers;
    use crate::chat::reader::ChatFile;
    use crate::chat::utterance::Utterance;
    use std::collections::HashMap;

    fn make_utterance(
        participant: &str,
        main_text: &str,
        time_marks: Option<(i64, i64)>,
    ) -> Utterance {
        let mut tiers = HashMap::new();
        tiers.insert(participant.to_string(), main_text.to_string());
        Utterance {
            participant: Some(participant.to_string()),
            tokens: None,
            time_marks,
            tiers: Some(tiers),
            changeable_header: None,
            mor_tier_name: Some("%mor".to_string()),
            gra_tier_name: Some("%gra".to_string()),
        }
    }

    fn make_chat_file(events: Vec<Utterance>) -> ChatFile {
        ChatFile::new("test.cha".to_string(), Headers::default(), events, vec![])
    }

    #[test]
    fn test_single_participant() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "hello world .", Some((0, 1500))),
            make_utterance("CHI", "goodbye .", Some((2000, 3500))),
        ]);
        let srt = chat_file_to_srt_str(&file, None);

        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhello world ."));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:03,500\ngoodbye ."));
        // No participant prefix for single participant.
        assert!(!srt.contains("CHI:"));
    }

    #[test]
    fn test_multi_participant() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "more cookie .", Some((0, 2000))),
            make_utterance("MOT", "want more ?", Some((2500, 4000))),
        ]);
        let srt = chat_file_to_srt_str(&file, None);

        // Participant prefix for multiple participants.
        assert!(srt.contains("CHI: more cookie ."));
        assert!(srt.contains("MOT: want more ?"));
    }

    #[test]
    fn test_skip_no_time_marks() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "hello .", Some((0, 1500))),
            make_utterance("CHI", "no time marks .", None),
            make_utterance("CHI", "goodbye .", Some((2000, 3000))),
        ]);
        let srt = chat_file_to_srt_str(&file, None);

        assert!(srt.contains("hello ."));
        assert!(!srt.contains("no time marks ."));
        assert!(srt.contains("goodbye ."));
        // Only 2 entries.
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
        assert!(!srt.contains("3\n"));
    }

    #[test]
    fn test_filter_participants() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "more cookie .", Some((0, 2000))),
            make_utterance("MOT", "want more ?", Some((2500, 4000))),
        ]);
        let participants = vec!["CHI".to_string()];
        let srt = chat_file_to_srt_str(&file, Some(&participants));

        assert!(srt.contains("more cookie ."));
        assert!(!srt.contains("want more ?"));
        // Single participant after filtering, no prefix.
        assert!(!srt.contains("CHI:"));
    }

    #[test]
    fn test_strip_bullet_markers() {
        let text = "hello world . \x150_1500\x15";
        let result = strip_bullet_markers(text);
        assert_eq!(result, "hello world .");
    }

    #[test]
    fn test_empty_file() {
        let file = make_chat_file(vec![]);
        let srt = chat_file_to_srt_str(&file, None);
        assert!(srt.is_empty());
    }

    #[test]
    fn test_sorted_by_time() {
        let file = make_chat_file(vec![
            make_utterance("CHI", "second .", Some((2000, 3000))),
            make_utterance("CHI", "first .", Some((0, 1000))),
        ]);
        let srt = chat_file_to_srt_str(&file, None);

        let first_pos = srt.find("first .").unwrap();
        let second_pos = srt.find("second .").unwrap();
        assert!(first_pos < second_pos);
    }
}
