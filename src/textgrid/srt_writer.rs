//! Convert TextGrid data to SRT (SubRip Subtitle) format.

use super::reader::{TextGridFile, TextGridTier};
use crate::srt::format_srt_time;

/// A single subtitle entry collected from TextGrid annotations.
struct SrtEntry {
    text: String,
    start_ms: i64,
    end_ms: i64,
}

/// Convert a single [`TextGridFile`] to an SRT format string.
///
/// If `participants` is `Some`, only IntervalTiers with those names are included.
/// If `None`, auto-detect: IntervalTiers with a 3-character name.
/// TextTiers are always skipped. Empty-text intervals are skipped.
/// When multiple tiers are selected, the subtitle text is prefixed with the tier name.
pub(crate) fn textgrid_file_to_srt_str(
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

    let multi = main_tier_names.len() > 1;

    // 2. Collect entries.
    let mut entries: Vec<SrtEntry> = Vec::new();
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
                let text = if multi {
                    format!("{}: {}", name, interval.text)
                } else {
                    interval.text.clone()
                };
                entries.push(SrtEntry {
                    text,
                    start_ms,
                    end_ms,
                });
            }
        }
    }

    // 3. Sort by start time, then end time.
    entries.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    // 4. Generate SRT output.
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
            vec![
                Interval {
                    xmin: 0.0,
                    xmax: 1.5,
                    text: "hello world .".to_string(),
                },
                Interval {
                    xmin: 2.0,
                    xmax: 3.5,
                    text: "goodbye .".to_string(),
                },
            ],
        )]);
        let srt = textgrid_file_to_srt_str(&file, None);

        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,500\nhello world ."));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:03,500\ngoodbye ."));
        // No prefix for single tier.
        assert!(!srt.contains("CHI:"));
    }

    #[test]
    fn test_multi_tier() {
        let file = make_textgrid_file(vec![
            make_interval_tier(
                "CHI",
                vec![Interval {
                    xmin: 0.0,
                    xmax: 2.0,
                    text: "more cookie .".to_string(),
                }],
            ),
            make_interval_tier(
                "MOT",
                vec![Interval {
                    xmin: 2.5,
                    xmax: 4.0,
                    text: "want more ?".to_string(),
                }],
            ),
        ]);
        let srt = textgrid_file_to_srt_str(&file, None);

        assert!(srt.contains("CHI: more cookie ."));
        assert!(srt.contains("MOT: want more ?"));
    }

    #[test]
    fn test_explicit_participants() {
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
        let participants = vec!["Speaker1".to_string()];
        let srt = textgrid_file_to_srt_str(&file, Some(&participants));

        assert!(srt.contains("hello"));
        assert!(!srt.contains("hi"));
        assert!(!srt.contains("Speaker1:"));
    }

    #[test]
    fn test_empty_file() {
        let file = make_textgrid_file(vec![]);
        let srt = textgrid_file_to_srt_str(&file, None);
        assert!(srt.is_empty());
    }
}
