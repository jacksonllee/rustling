//! Convert SRT (SubRip Subtitle) data to TextGrid (Praat) format.

use super::reader::SrtFile;
use crate::textgrid::serialize_textgrid_file;

/// Convert a single [`SrtFile`] to a TextGrid format string.
///
/// Creates a single IntervalTier named `"SPK"` (Speaker).
pub(crate) fn srt_file_to_textgrid_str(file: &SrtFile) -> String {
    let mut intervals: Vec<crate::textgrid::Interval> = Vec::new();
    let mut global_xmax: f64 = 0.0;

    for block in &file.blocks {
        let xmin = block.start_ms as f64 / 1000.0;
        let xmax = block.end_ms as f64 / 1000.0;
        if xmax > global_xmax {
            global_xmax = xmax;
        }
        // Join multiline subtitle text with space.
        let text = block.text.replace('\n', " ");
        intervals.push(crate::textgrid::Interval { xmin, xmax, text });
    }

    let tiers = vec![crate::textgrid::TextGridTier::IntervalTier {
        name: "SPK".to_string(),
        xmin: 0.0,
        xmax: global_xmax,
        intervals,
    }];

    let tg_file = crate::textgrid::TextGridFile {
        file_path: file.file_path.clone(),
        xmin: 0.0,
        xmax: global_xmax,
        tiers,
        raw_text: String::new(),
    };
    serialize_textgrid_file(&tg_file)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::srt::{SrtBlock, SrtFile};
    use crate::textgrid::{BaseTextGrid, TextGrid};

    #[test]
    fn test_basic() {
        let file = SrtFile {
            file_path: "test.srt".to_string(),
            blocks: vec![
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
            ],
        };
        let tg_str = srt_file_to_textgrid_str(&file);

        assert!(tg_str.contains("\"IntervalTier\""));
        assert!(tg_str.contains("\"SPK\""));
        assert!(tg_str.contains("\"Hello world.\""));
        assert!(tg_str.contains("\"Goodbye.\""));
    }

    #[test]
    fn test_multiline_joined() {
        let file = SrtFile {
            file_path: "test.srt".to_string(),
            blocks: vec![SrtBlock {
                index: 1,
                text: "Line one.\nLine two.".to_string(),
                start_ms: 0,
                end_ms: 2000,
            }],
        };
        let tg_str = srt_file_to_textgrid_str(&file);

        assert!(tg_str.contains("\"Line one. Line two.\""));
    }

    #[test]
    fn test_round_trip() {
        let file = SrtFile {
            file_path: "test.srt".to_string(),
            blocks: vec![SrtBlock {
                index: 1,
                text: "Hello.".to_string(),
                start_ms: 0,
                end_ms: 1500,
            }],
        };
        let tg_str = srt_file_to_textgrid_str(&file);

        let tg = TextGrid::from_strs(vec![tg_str], Some(vec!["test.TextGrid".to_string()]), false)
            .unwrap();
        assert_eq!(tg.tiers_flat().len(), 1);
        assert_eq!(tg.tiers_flat()[0].name(), "SPK");
    }

    #[test]
    fn test_empty() {
        let file = SrtFile {
            file_path: "test.srt".to_string(),
            blocks: vec![],
        };
        let tg_str = srt_file_to_textgrid_str(&file);
        assert!(tg_str.contains("intervals: size = 0"));
    }
}
