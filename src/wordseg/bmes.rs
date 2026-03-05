//! Helpers for shared behavior across word segmentation types.

// ---------------------------------------------------------------------------
// BMES conversion helpers
// ---------------------------------------------------------------------------

/// Convert segmented words into character-level BMES labels.
///
/// - Single-char word → S
/// - Multi-char word: first char → B, middle chars → M, last char → E
pub(crate) fn words_to_bmes(words: &[String]) -> Vec<(char, &'static str)> {
    let mut result = Vec::new();
    for word in words {
        let chars: Vec<char> = word.chars().collect();
        match chars.len() {
            0 => {}
            1 => result.push((chars[0], "S")),
            n => {
                result.push((chars[0], "B"));
                for &c in &chars[1..n - 1] {
                    result.push((c, "M"));
                }
                result.push((chars[n - 1], "E"));
            }
        }
    }
    result
}

/// Convert BMES-labeled characters back into segmented words.
///
/// Handles invalid transitions gracefully:
/// - M or E without a preceding B: treated as starting a new word.
/// - B followed by B or S: emits the current partial word first.
pub(crate) fn bmes_to_words(chars: &[char], tags: &[&str]) -> Vec<String> {
    let mut words = Vec::new();
    let mut current_word = String::new();

    for (&ch, &tag) in chars.iter().zip(tags.iter()) {
        match tag {
            "S" => {
                if !current_word.is_empty() {
                    words.push(std::mem::take(&mut current_word));
                }
                words.push(ch.to_string());
            }
            "B" => {
                if !current_word.is_empty() {
                    words.push(std::mem::take(&mut current_word));
                }
                current_word.push(ch);
            }
            "M" => {
                current_word.push(ch);
            }
            "E" => {
                if current_word.is_empty() {
                    words.push(ch.to_string());
                } else {
                    current_word.push(ch);
                    words.push(std::mem::take(&mut current_word));
                }
            }
            _ => {
                if !current_word.is_empty() {
                    words.push(std::mem::take(&mut current_word));
                }
                words.push(ch.to_string());
            }
        }
    }

    if !current_word.is_empty() {
        words.push(current_word);
    }

    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    // ---- words_to_bmes ----

    #[test]
    fn words_to_bmes_single_char_word() {
        let result = words_to_bmes(&s(&["a"]));
        assert_eq!(result, vec![('a', "S")]);
    }

    #[test]
    fn words_to_bmes_two_char_word() {
        let result = words_to_bmes(&s(&["hi"]));
        assert_eq!(result, vec![('h', "B"), ('i', "E")]);
    }

    #[test]
    fn words_to_bmes_multi_char_word() {
        let result = words_to_bmes(&s(&["hello"]));
        assert_eq!(
            result,
            vec![('h', "B"), ('e', "M"), ('l', "M"), ('l', "M"), ('o', "E"),]
        );
    }

    #[test]
    fn words_to_bmes_mixed_words() {
        let result = words_to_bmes(&s(&["I", "am", "good"]));
        assert_eq!(
            result,
            vec![
                ('I', "S"),
                ('a', "B"),
                ('m', "E"),
                ('g', "B"),
                ('o', "M"),
                ('o', "M"),
                ('d', "E"),
            ]
        );
    }

    #[test]
    fn words_to_bmes_empty_input() {
        let result = words_to_bmes(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn words_to_bmes_empty_string_skipped() {
        let result = words_to_bmes(&s(&["a", "", "b"]));
        assert_eq!(result, vec![('a', "S"), ('b', "S")]);
    }

    #[test]
    fn words_to_bmes_cjk() {
        // 今天 = two chars, 好 = one char
        let result = words_to_bmes(&s(&["今天", "好"]));
        assert_eq!(result, vec![('今', "B"), ('天', "E"), ('好', "S")]);
    }

    // ---- bmes_to_words ----

    #[test]
    fn bmes_to_words_single_s() {
        let result = bmes_to_words(&['a'], &["S"]);
        assert_eq!(result, vec!["a"]);
    }

    #[test]
    fn bmes_to_words_be() {
        let result = bmes_to_words(&['h', 'i'], &["B", "E"]);
        assert_eq!(result, vec!["hi"]);
    }

    #[test]
    fn bmes_to_words_bme() {
        let result = bmes_to_words(&['c', 'a', 't'], &["B", "M", "E"]);
        assert_eq!(result, vec!["cat"]);
    }

    #[test]
    fn bmes_to_words_mixed() {
        let chars: Vec<char> = "Iamgood".chars().collect();
        let tags = vec!["S", "B", "E", "B", "M", "M", "E"];
        let result = bmes_to_words(&chars, &tags);
        assert_eq!(result, vec!["I", "am", "good"]);
    }

    #[test]
    fn bmes_to_words_empty() {
        let result = bmes_to_words(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn bmes_to_words_e_without_b() {
        // E without preceding B → treated as standalone word
        let result = bmes_to_words(&['x'], &["E"]);
        assert_eq!(result, vec!["x"]);
    }

    #[test]
    fn bmes_to_words_b_followed_by_b() {
        // Second B flushes the partial word from the first B
        let result = bmes_to_words(&['a', 'b', 'c'], &["B", "B", "E"]);
        assert_eq!(result, vec!["a", "bc"]);
    }

    #[test]
    fn bmes_to_words_b_followed_by_s() {
        // S flushes the partial word from the preceding B
        let result = bmes_to_words(&['a', 'b'], &["B", "S"]);
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn bmes_to_words_trailing_b() {
        // B at end with no E → emitted as partial word
        let result = bmes_to_words(&['a', 'b'], &["S", "B"]);
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn bmes_to_words_unknown_tag() {
        let result = bmes_to_words(&['a', 'b', 'c'], &["S", "X", "S"]);
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    // ---- roundtrip ----

    #[test]
    fn roundtrip() {
        let words = s(&["I", "am", "good"]);
        let bmes = words_to_bmes(&words);
        let chars: Vec<char> = bmes.iter().map(|(c, _)| *c).collect();
        let tags: Vec<&str> = bmes.iter().map(|(_, t)| *t).collect();
        let recovered = bmes_to_words(&chars, &tags);
        assert_eq!(recovered, words);
    }

    #[test]
    fn roundtrip_cjk() {
        let words = s(&["今天", "天气", "好"]);
        let bmes = words_to_bmes(&words);
        let chars: Vec<char> = bmes.iter().map(|(c, _)| *c).collect();
        let tags: Vec<&str> = bmes.iter().map(|(_, t)| *t).collect();
        let recovered = bmes_to_words(&chars, &tags);
        assert_eq!(recovered, words);
    }
}
