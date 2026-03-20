//! Convert TextGrid data to CHAT format.

use super::reader::{TextGridFile, TextGridTier};

/// A single utterance collected from TextGrid annotations for CHAT output.
struct ChatUtterance {
    participant: String,
    text: String,
    start_ms: i64,
    end_ms: i64,
}

/// Convert a single [`TextGridFile`] to a CHAT format string.
///
/// If `participants` is `Some`, only IntervalTiers with those names are included.
/// If `None`, auto-detect: IntervalTiers with a 3-character name.
/// TextTiers are always skipped.
pub(crate) fn textgrid_file_to_chat_str(
    file: &TextGridFile,
    participants: Option<&[String]>,
) -> String {
    // 1. Identify main IntervalTier names.
    let main_tier_names: Vec<&str> = file
        .tiers
        .iter()
        .filter(|t| {
            matches!(t, TextGridTier::IntervalTier { .. })
                && match participants {
                    Some(codes) => codes.iter().any(|c| c == t.name()),
                    None => t.name().len() == 3,
                }
        })
        .map(|t| t.name())
        .collect();

    // 2. Collect utterances from selected tiers.
    let mut utterances: Vec<ChatUtterance> = Vec::new();
    for tier in &file.tiers {
        if let TextGridTier::IntervalTier {
            name, intervals, ..
        } = tier
        {
            if !main_tier_names.contains(&name.as_str()) {
                continue;
            }
            for interval in intervals {
                if interval.text.is_empty() {
                    continue;
                }
                let start_ms = (interval.xmin * 1000.0).round() as i64;
                let end_ms = (interval.xmax * 1000.0).round() as i64;
                utterances.push(ChatUtterance {
                    participant: name.clone(),
                    text: interval.text.clone(),
                    start_ms,
                    end_ms,
                });
            }
        }
    }

    // 3. Sort by (start_time, end_time).
    utterances.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    // 4. Generate CHAT output.
    let mut output = String::with_capacity(4096);
    output.push_str("@UTF8\n");
    output.push_str("@Begin\n");

    // @Participants line.
    if !main_tier_names.is_empty() {
        output.push_str("@Participants:\t");
        let parts: Vec<String> = main_tier_names
            .iter()
            .map(|code| format!("{code} {code}"))
            .collect();
        output.push_str(&parts.join(", "));
        output.push('\n');
    }

    // Utterances.
    for utt in &utterances {
        output.push_str(&format!(
            "*{}:\t{} \x15{}_{}\x15\n",
            utt.participant, utt.text, utt.start_ms, utt.end_ms,
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
    use crate::textgrid::reader::{Interval, TextGridFile};

    fn make_interval_tier(name: &str, intervals: Vec<Interval>) -> TextGridTier {
        let xmax = intervals.last().map(|i| i.xmax).unwrap_or(0.0);
        TextGridTier::IntervalTier {
            name: name.to_string(),
            xmin: 0.0,
            xmax,
            intervals,
        }
    }

    fn make_textgrid_file(tiers: Vec<TextGridTier>) -> TextGridFile {
        TextGridFile {
            file_path: "test.TextGrid".to_string(),
            xmin: 0.0,
            xmax: 5.0,
            tiers,
            raw_text: String::new(),
        }
    }

    #[test]
    fn test_single_tier() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "CHI",
            vec![Interval {
                xmin: 0.0,
                xmax: 1.5,
                text: "hello world .".to_string(),
            }],
        )]);
        let chat = textgrid_file_to_chat_str(&file, None);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
        assert!(chat.contains("@Participants:\tCHI CHI"));
        assert!(chat.contains("*CHI:\thello world . \x150_1500\x15"));
    }

    #[test]
    fn test_auto_detect_skips_non_3char() {
        let file = make_textgrid_file(vec![
            make_interval_tier(
                "Speaker1",
                vec![Interval {
                    xmin: 0.0,
                    xmax: 1.0,
                    text: "hello".to_string(),
                }],
            ),
            make_interval_tier(
                "CHI",
                vec![Interval {
                    xmin: 1.0,
                    xmax: 2.0,
                    text: "hi".to_string(),
                }],
            ),
        ]);
        let chat = textgrid_file_to_chat_str(&file, None);

        assert!(chat.contains("*CHI:\thi"));
        assert!(!chat.contains("Speaker1"));
        assert!(!chat.contains("hello"));
    }

    #[test]
    fn test_explicit_participants() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "Speaker1",
            vec![Interval {
                xmin: 0.0,
                xmax: 1.0,
                text: "hello".to_string(),
            }],
        )]);
        let participants = vec!["Speaker1".to_string()];
        let chat = textgrid_file_to_chat_str(&file, Some(&participants));

        assert!(chat.contains("*Speaker1:\thello"));
    }

    #[test]
    fn test_skips_empty_intervals() {
        let file = make_textgrid_file(vec![make_interval_tier(
            "CHI",
            vec![
                Interval {
                    xmin: 0.0,
                    xmax: 0.5,
                    text: String::new(),
                },
                Interval {
                    xmin: 0.5,
                    xmax: 1.0,
                    text: "hello .".to_string(),
                },
            ],
        )]);
        let chat = textgrid_file_to_chat_str(&file, None);

        assert!(chat.contains("hello ."));
        assert_eq!(chat.matches("*CHI:").count(), 1);
    }

    #[test]
    fn test_empty_file() {
        let file = make_textgrid_file(vec![]);
        let chat = textgrid_file_to_chat_str(&file, None);

        assert!(chat.contains("@Begin"));
        assert!(chat.contains("@End"));
        assert!(!chat.contains("@Participants"));
    }
}
