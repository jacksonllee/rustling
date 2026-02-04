//! Trie (prefix tree) data structure for efficient sequence matching.
//!
//! A trie is a tree-like data structure that stores sequences by their prefixes,
//! enabling efficient operations like prefix matching, longest match finding,
//! and membership testing.
//!
//! The trie is generic over the element type, making it suitable for:
//! - Characters in words
//! - Words in sentences
//! - Phonemes in transcriptions
//! - Any other sequential data
//!
//! # Example
//!
//! ```
//! use rustling::trie::Trie;
//!
//! // Character-based trie for words
//! let mut trie: Trie<char> = Trie::new();
//! trie.insert("hello".chars());
//! trie.insert("help".chars());
//!
//! assert!(trie.contains("hello".chars()));
//! assert!(!trie.contains("hell".chars()));
//!
//! let chars: Vec<char> = "helloworld".chars().collect();
//! assert_eq!(trie.longest_match(&chars, 10), 5); // "hello"
//!
//! // Phoneme-based trie (using strings as phoneme symbols)
//! let mut phoneme_trie: Trie<&str> = Trie::new();
//! phoneme_trie.insert(["h", "ɛ", "l", "oʊ"].iter().copied());
//! phoneme_trie.insert(["w", "ɜː", "l", "d"].iter().copied());
//!
//! assert!(phoneme_trie.contains(["h", "ɛ", "l", "oʊ"].iter().copied()));
//! ```

use std::collections::HashMap;
use std::hash::Hash;

/// A trie node containing children and a flag indicating if this node ends a sequence.
#[derive(Clone, Debug)]
pub struct TrieNode<T: Eq + Hash + Clone> {
    children: HashMap<T, TrieNode<T>>,
    is_terminal: bool,
}

impl<T: Eq + Hash + Clone> Default for TrieNode<T> {
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            is_terminal: false,
        }
    }
}

impl<T: Eq + Hash + Clone> TrieNode<T> {
    /// Create a new empty trie node.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if this node has any children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Get the number of children.
    pub fn num_children(&self) -> usize {
        self.children.len()
    }
}

/// A trie (prefix tree) for efficient sequence operations.
///
/// This implementation is generic over the element type `T`, which must implement
/// `Eq`, `Hash`, and `Clone`. This allows the trie to store sequences of any type:
/// characters, phonemes, tokens, etc.
///
/// This implementation supports:
/// - Insertion of sequences
/// - Membership testing
/// - Longest prefix matching
/// - Prefix existence checking
#[derive(Clone, Debug)]
pub struct Trie<T: Eq + Hash + Clone> {
    root: TrieNode<T>,
    len: usize,
}

impl<T: Eq + Hash + Clone> Default for Trie<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash + Clone> Trie<T> {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            len: 0,
        }
    }

    /// Insert a sequence into the trie.
    ///
    /// Returns `true` if the sequence was newly inserted, `false` if it already existed.
    pub fn insert<I>(&mut self, sequence: I) -> bool
    where
        I: IntoIterator<Item = T>,
    {
        let mut node = &mut self.root;
        for element in sequence {
            node = node.children.entry(element).or_default();
        }
        if node.is_terminal {
            false
        } else {
            node.is_terminal = true;
            self.len += 1;
            true
        }
    }

    /// Check if the trie contains the exact sequence.
    pub fn contains<I>(&self, sequence: I) -> bool
    where
        I: IntoIterator<Item = T>,
    {
        let mut node = &self.root;
        for element in sequence {
            match node.children.get(&element) {
                Some(child) => node = child,
                None => return false,
            }
        }
        node.is_terminal
    }

    /// Check if the trie contains any sequence with the given prefix.
    pub fn has_prefix<I>(&self, prefix: I) -> bool
    where
        I: IntoIterator<Item = T>,
    {
        let mut node = &self.root;
        for element in prefix {
            match node.children.get(&element) {
                Some(child) => node = child,
                None => return false,
            }
        }
        true
    }

    /// Find the longest matching sequence starting at the beginning of the given slice.
    ///
    /// Returns the length of the longest match, or 0 if no match.
    ///
    /// # Arguments
    ///
    /// * `elements` - The slice of elements to match against.
    /// * `max_len` - Maximum number of elements to consider.
    pub fn longest_match(&self, elements: &[T], max_len: usize) -> usize {
        let mut node = &self.root;
        let mut longest = 0;

        for (i, element) in elements.iter().take(max_len).enumerate() {
            match node.children.get(element) {
                Some(child) => {
                    node = child;
                    if node.is_terminal {
                        longest = i + 1;
                    }
                }
                None => break,
            }
        }

        longest
    }

    /// Find the longest matching sequence from an iterator.
    ///
    /// This is a convenience method that collects the iterator into a Vec internally.
    /// For better performance with slices, use `longest_match` directly.
    ///
    /// # Arguments
    ///
    /// * `sequence` - An iterator of elements to match against.
    /// * `max_len` - Maximum number of elements to consider.
    pub fn longest_match_iter<I>(&self, sequence: I, max_len: usize) -> usize
    where
        I: IntoIterator<Item = T>,
    {
        let elements: Vec<T> = sequence.into_iter().take(max_len).collect();
        self.longest_match(&elements, max_len)
    }

    /// Get the number of sequences in the trie.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all sequences from the trie.
    pub fn clear(&mut self) {
        self.root = TrieNode::new();
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_trie_is_empty() {
        let trie: Trie<char> = Trie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut trie: Trie<char> = Trie::new();
        assert!(trie.insert("hello".chars()));
        assert!(trie.insert("world".chars()));
        assert!(!trie.insert("hello".chars())); // duplicate

        assert!(trie.contains("hello".chars()));
        assert!(trie.contains("world".chars()));
        assert!(!trie.contains("hell".chars()));
        assert!(!trie.contains("hello!".chars()));
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_has_prefix() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("hello".chars());
        trie.insert("help".chars());

        assert!(trie.has_prefix("hel".chars()));
        assert!(trie.has_prefix("hello".chars()));
        assert!(trie.has_prefix("help".chars()));
        assert!(!trie.has_prefix("hex".chars()));
        assert!(!trie.has_prefix("world".chars()));
    }

    #[test]
    fn test_longest_match() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("he".chars());
        trie.insert("hello".chars());
        trie.insert("help".chars());

        let chars: Vec<char> = "helloworld".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 5); // "hello"

        let chars: Vec<char> = "helping".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 4); // "help"

        let chars: Vec<char> = "hex".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 2); // "he"

        let chars: Vec<char> = "world".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 0); // no match
    }

    #[test]
    fn test_longest_match_with_max_len() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("hello".chars());

        let chars: Vec<char> = "helloworld".chars().collect();
        assert_eq!(trie.longest_match(&chars, 3), 0); // max_len too short
        assert_eq!(trie.longest_match(&chars, 5), 5); // exactly matches
        assert_eq!(trie.longest_match(&chars, 10), 5); // longer than needed
    }

    #[test]
    fn test_unicode() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("你好".chars());
        trie.insert("世界".chars());
        trie.insert("你好世界".chars());

        assert!(trie.contains("你好".chars()));
        assert!(trie.contains("世界".chars()));
        assert!(!trie.contains("你".chars()));

        let chars: Vec<char> = "你好世界".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 4); // "你好世界"
    }

    #[test]
    fn test_clear() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("hello".chars());
        trie.insert("world".chars());
        assert_eq!(trie.len(), 2);

        trie.clear();
        assert!(trie.is_empty());
        assert!(!trie.contains("hello".chars()));
    }

    #[test]
    fn test_longest_match_iter() {
        let mut trie: Trie<char> = Trie::new();
        trie.insert("hello".chars());

        assert_eq!(trie.longest_match_iter("helloworld".chars(), 10), 5);
        assert_eq!(trie.longest_match_iter("world".chars(), 10), 0);
    }

    #[test]
    fn test_phoneme_trie() {
        // Demonstrate using the trie with phoneme symbols (strings)
        let mut trie: Trie<&str> = Trie::new();

        // Insert phoneme sequences
        let hello_phonemes = ["h", "ə", "l", "oʊ"];
        let help_phonemes = ["h", "ɛ", "l", "p"];
        let world_phonemes = ["w", "ɜː", "l", "d"];

        trie.insert(hello_phonemes.iter().copied());
        trie.insert(help_phonemes.iter().copied());
        trie.insert(world_phonemes.iter().copied());

        assert!(trie.contains(hello_phonemes.iter().copied()));
        assert!(trie.contains(help_phonemes.iter().copied()));
        assert!(!trie.contains(["h", "ə"].iter().copied())); // prefix only

        // Test longest match with phonemes
        let test_phonemes = ["h", "ə", "l", "oʊ", "w", "ɜː", "l", "d"];
        assert_eq!(trie.longest_match(&test_phonemes, 10), 4); // matches "hello"
    }

    #[test]
    fn test_integer_trie() {
        // Demonstrate using the trie with integers (e.g., token IDs)
        let mut trie: Trie<u32> = Trie::new();

        trie.insert([1, 2, 3].iter().copied());
        trie.insert([1, 2, 4].iter().copied());
        trie.insert([5, 6, 7].iter().copied());

        assert!(trie.contains([1, 2, 3].iter().copied()));
        assert!(!trie.contains([1, 2].iter().copied()));

        let sequence = [1, 2, 3, 5, 6, 7];
        assert_eq!(trie.longest_match(&sequence, 10), 3);
    }
}
