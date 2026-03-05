//! Trie (prefix tree) data structure for efficient sequence matching.
//!
//! A generic trie where each node has children (trie edges) and an optional
//! terminal marker of type `V`. The terminal marker indicates that a sequence
//! was explicitly inserted at that node and can carry associated data.
//!
//! The trie is generic over the key type `K` and terminal value type `V`:
//! - `Trie<K, ()>` for membership testing (terminal marker is just a flag)
//! - `Trie<K, u64>` for counting (terminal marker carries a count)
//!
//! # Examples
//!
//! Membership trie:
//!
//! ```
//! use rustling::trie::Trie;
//!
//! let mut trie: Trie<char, ()> = Trie::new();
//! trie.insert_seq("hello".chars());
//! trie.insert_seq("help".chars());
//!
//! assert!(trie.contains("hello".chars()));
//! assert!(!trie.contains("hell".chars()));
//!
//! let chars: Vec<char> = "helloworld".chars().collect();
//! assert_eq!(trie.longest_match(&chars, 10), 5); // "hello"
//! ```
//!
//! Counting trie:
//!
//! ```
//! use rustling::trie::CountTrie;
//!
//! let mut trie: CountTrie<String> = CountTrie::new();
//! trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
//! trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
//! trie.increment(["the", "dog"].iter().map(|s| s.to_string()));
//!
//! assert_eq!(trie.get_count(["the", "cat"].iter().map(|s| s.to_string())), 2);
//! assert_eq!(trie.get_count(["the", "dog"].iter().map(|s| s.to_string())), 1);
//! assert_eq!(trie.children_count_sum(std::iter::once("the".to_string())), 3);
//! ```

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

/// A trie node containing children and an optional terminal marker.
///
/// The terminal marker (`Option<V>`) indicates whether a sequence was
/// explicitly inserted ending at this node:
/// - `None` — this node exists only as a prefix (structural)
/// - `Some(v)` — a sequence was explicitly inserted here, carrying value `v`
///
/// A node can be both terminal and have children (e.g., "the" is a word
/// but also a prefix of "them").
#[derive(Clone, Debug)]
pub struct TrieNode<K: Eq + Hash + Clone, V> {
    children: HashMap<K, TrieNode<K, V>>,
    terminal: Option<V>,
}

impl<K: Eq + Hash + Clone, V> Default for TrieNode<K, V> {
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            terminal: None,
        }
    }
}

impl<K: Eq + Hash + Clone, V> TrieNode<K, V> {
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

    /// Get the terminal value, if present.
    pub fn terminal(&self) -> Option<&V> {
        self.terminal.as_ref()
    }

    /// Get the child node for the given key, if present.
    pub fn get_child(&self, key: &K) -> Option<&TrieNode<K, V>> {
        self.children.get(key)
    }
}

/// A trie (prefix tree) for efficient sequence operations.
///
/// Generic over the key type `K` and terminal value type `V`.
/// Use `Trie<K, ()>` for membership testing, `Trie<K, u64>` for counting.
#[derive(Clone, Debug)]
pub struct Trie<K: Eq + Hash + Clone, V> {
    root: TrieNode<K, V>,
    len: usize,
}

impl<K: Eq + Hash + Clone, V> Default for Trie<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone, V> Trie<K, V> {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            len: 0,
        }
    }

    /// Insert a sequence with the given terminal value.
    ///
    /// Returns the previous terminal value if the sequence was already present.
    pub fn insert<I>(&mut self, sequence: I, value: V) -> Option<V>
    where
        I: IntoIterator<Item = K>,
    {
        let mut node = &mut self.root;
        for element in sequence {
            node = node.children.entry(element).or_default();
        }
        let old = node.terminal.take();
        node.terminal = Some(value);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    /// Get a reference to the terminal value for the given sequence.
    ///
    /// Returns `None` if the sequence is not present or has no terminal marker.
    pub fn get<I>(&self, sequence: I) -> Option<&V>
    where
        I: IntoIterator<Item = K>,
    {
        let mut node = &self.root;
        for element in sequence {
            match node.children.get(&element) {
                Some(child) => node = child,
                None => return None,
            }
        }
        node.terminal.as_ref()
    }

    /// Check if the trie contains the exact sequence (has a terminal marker).
    pub fn contains<I>(&self, sequence: I) -> bool
    where
        I: IntoIterator<Item = K>,
    {
        self.get(sequence).is_some()
    }

    /// Check if the trie contains any node at the given prefix path.
    pub fn has_prefix<I>(&self, prefix: I) -> bool
    where
        I: IntoIterator<Item = K>,
    {
        self.get_node(prefix).is_some()
    }

    /// Navigate to the node at the given sequence path.
    ///
    /// Returns `None` if the path does not exist in the trie.
    pub fn get_node<I>(&self, sequence: I) -> Option<&TrieNode<K, V>>
    where
        I: IntoIterator<Item = K>,
    {
        let mut node = &self.root;
        for element in sequence {
            match node.children.get(&element) {
                Some(child) => node = child,
                None => return None,
            }
        }
        Some(node)
    }

    /// Get direct children that have terminal markers.
    ///
    /// Returns a vector of `(key, &value)` pairs. Returns an empty vector
    /// if the context path is not found.
    pub fn children<I>(&self, context: I) -> Vec<(K, &V)>
    where
        I: IntoIterator<Item = K>,
    {
        match self.get_node(context) {
            Some(node) => node
                .children
                .iter()
                .filter_map(|(k, v)| v.terminal.as_ref().map(|t| (k.clone(), t)))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get the number of sequences in the trie (nodes with terminal markers).
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

// --- Membership convenience methods for Trie<K, ()> ---

impl<K: Eq + Hash + Clone> Trie<K, ()> {
    /// Insert a sequence (membership only).
    ///
    /// Returns `true` if the sequence was newly inserted, `false` if it already existed.
    pub fn insert_seq<I>(&mut self, sequence: I) -> bool
    where
        I: IntoIterator<Item = K>,
    {
        self.insert(sequence, ()).is_none()
    }

    /// Find the longest matching sequence starting at the beginning of the given slice.
    ///
    /// Returns the length of the longest match, or 0 if no match.
    ///
    /// # Arguments
    ///
    /// * `elements` - The slice of elements to match against.
    /// * `max_len` - Maximum number of elements to consider.
    pub fn longest_match(&self, elements: &[K], max_len: usize) -> usize {
        let mut node = &self.root;
        let mut longest = 0;

        for (i, element) in elements.iter().take(max_len).enumerate() {
            match node.children.get(element) {
                Some(child) => {
                    node = child;
                    if node.terminal.is_some() {
                        longest = i + 1;
                    }
                }
                None => break,
            }
        }

        longest
    }

    /// Collect all terminal sequences in the trie.
    ///
    /// Returns a vector of all sequences that were explicitly inserted.
    /// The order of entries is not guaranteed.
    pub fn all_sequences(&self) -> Vec<Vec<K>> {
        let mut result = Vec::with_capacity(self.len());
        let mut prefix = Vec::new();
        Self::collect_sequences(&self.root, &mut prefix, &mut result);
        result
    }

    fn collect_sequences(node: &TrieNode<K, ()>, prefix: &mut Vec<K>, result: &mut Vec<Vec<K>>) {
        if node.terminal.is_some() {
            result.push(prefix.clone());
        }
        for (key, child) in &node.children {
            prefix.push(key.clone());
            Self::collect_sequences(child, prefix, result);
            prefix.pop();
        }
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
        I: IntoIterator<Item = K>,
    {
        let elements: Vec<K> = sequence.into_iter().take(max_len).collect();
        self.longest_match(&elements, max_len)
    }
}

// --- CountTrie: a counting trie wrapping Trie<K, u64> ---

/// A counting trie for efficient n-gram frequency storage.
///
/// Wraps `Trie<K, u64>` and adds counting-specific methods.
/// All `Trie` methods are available through `Deref`/`DerefMut`.
#[derive(Clone, Debug)]
pub struct CountTrie<K: Eq + Hash + Clone> {
    inner: Trie<K, u64>,
}

impl<K: Eq + Hash + Clone> Default for CountTrie<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone> Deref for CountTrie<K> {
    type Target = Trie<K, u64>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K: Eq + Hash + Clone> DerefMut for CountTrie<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<K: Eq + Hash + Clone> CountTrie<K> {
    /// Create a new empty counting trie.
    pub fn new() -> Self {
        Self { inner: Trie::new() }
    }

    /// Increment the count for the given sequence by 1.
    ///
    /// Creates intermediate nodes as needed. Sets count to 1 on first insert,
    /// increments on subsequent calls.
    pub fn increment<I>(&mut self, sequence: I)
    where
        I: IntoIterator<Item = K>,
    {
        let inner = &mut self.inner;
        let mut node = &mut inner.root;
        for element in sequence {
            node = node.children.entry(element).or_default();
        }
        match &mut node.terminal {
            Some(count) => *count += 1,
            None => {
                node.terminal = Some(1);
                inner.len += 1;
            }
        }
    }

    /// Set the count for the given sequence directly.
    ///
    /// Creates intermediate nodes as needed. Overwrites any existing count.
    /// This is more efficient than calling `increment` in a loop when
    /// loading counts from saved data.
    pub fn insert_count<I>(&mut self, sequence: I, count: u64)
    where
        I: IntoIterator<Item = K>,
    {
        let inner = &mut self.inner;
        let mut node = &mut inner.root;
        for element in sequence {
            node = node.children.entry(element).or_default();
        }
        if node.terminal.is_none() {
            inner.len += 1;
        }
        node.terminal = Some(count);
    }

    /// Get the count for the given sequence.
    ///
    /// Returns 0 if the sequence is not found.
    pub fn get_count<I>(&self, sequence: I) -> u64
    where
        I: IntoIterator<Item = K>,
    {
        self.inner.get(sequence).copied().unwrap_or(0)
    }

    /// Get the sum of counts of all direct children at the given context.
    ///
    /// This is the total number of observations following the context,
    /// used as the denominator in conditional probability calculations.
    ///
    /// Returns 0 if the context is not found.
    pub fn children_count_sum<I>(&self, context: I) -> u64
    where
        I: IntoIterator<Item = K>,
    {
        match self.inner.get_node(context) {
            Some(node) => node
                .children
                .values()
                .filter_map(|child| child.terminal)
                .sum(),
            None => 0,
        }
    }

    /// Get all direct children with their counts.
    ///
    /// Returns a vector of `(key, count)` pairs. Only includes children
    /// that have terminal markers. Returns an empty vector if the context
    /// is not found.
    pub fn children_with_counts<I>(&self, context: I) -> Vec<(K, u64)>
    where
        I: IntoIterator<Item = K>,
    {
        match self.inner.get_node(context) {
            Some(node) => node
                .children
                .iter()
                .filter_map(|(k, v)| v.terminal.map(|count| (k.clone(), count)))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Collect all sequences with their counts.
    ///
    /// Returns a vector of `(sequence, count)` pairs by recursively walking
    /// the trie. The order of entries is not guaranteed.
    pub fn all_counts(&self) -> Vec<(Vec<K>, u64)> {
        let mut result = Vec::with_capacity(self.len());
        let mut prefix = Vec::new();
        Self::collect_counts(&self.inner.root, &mut prefix, &mut result);
        result
    }

    /// Sum of all counts in the trie.
    pub fn total_count(&self) -> u64 {
        Self::sum_counts(&self.inner.root)
    }

    fn collect_counts(
        node: &TrieNode<K, u64>,
        prefix: &mut Vec<K>,
        result: &mut Vec<(Vec<K>, u64)>,
    ) {
        if let Some(count) = node.terminal {
            result.push((prefix.clone(), count));
        }
        for (key, child) in &node.children {
            prefix.push(key.clone());
            Self::collect_counts(child, prefix, result);
            prefix.pop();
        }
    }

    fn sum_counts(node: &TrieNode<K, u64>) -> u64 {
        let mut total = node.terminal.unwrap_or(0);
        for child in node.children.values() {
            total += Self::sum_counts(child);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Membership trie tests ---

    #[test]
    fn test_new_trie_is_empty() {
        let trie: Trie<char, ()> = Trie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut trie: Trie<char, ()> = Trie::new();
        assert!(trie.insert_seq("hello".chars()));
        assert!(trie.insert_seq("world".chars()));
        assert!(!trie.insert_seq("hello".chars())); // duplicate

        assert!(trie.contains("hello".chars()));
        assert!(trie.contains("world".chars()));
        assert!(!trie.contains("hell".chars()));
        assert!(!trie.contains("hello!".chars()));
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_has_prefix() {
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("hello".chars());
        trie.insert_seq("help".chars());

        assert!(trie.has_prefix("hel".chars()));
        assert!(trie.has_prefix("hello".chars()));
        assert!(trie.has_prefix("help".chars()));
        assert!(!trie.has_prefix("hex".chars()));
        assert!(!trie.has_prefix("world".chars()));
    }

    #[test]
    fn test_longest_match() {
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("he".chars());
        trie.insert_seq("hello".chars());
        trie.insert_seq("help".chars());

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
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("hello".chars());

        let chars: Vec<char> = "helloworld".chars().collect();
        assert_eq!(trie.longest_match(&chars, 3), 0); // max_len too short
        assert_eq!(trie.longest_match(&chars, 5), 5); // exactly matches
        assert_eq!(trie.longest_match(&chars, 10), 5); // longer than needed
    }

    #[test]
    fn test_unicode() {
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("你好".chars());
        trie.insert_seq("世界".chars());
        trie.insert_seq("你好世界".chars());

        assert!(trie.contains("你好".chars()));
        assert!(trie.contains("世界".chars()));
        assert!(!trie.contains("你".chars()));

        let chars: Vec<char> = "你好世界".chars().collect();
        assert_eq!(trie.longest_match(&chars, 10), 4); // "你好世界"
    }

    #[test]
    fn test_clear() {
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("hello".chars());
        trie.insert_seq("world".chars());
        assert_eq!(trie.len(), 2);

        trie.clear();
        assert!(trie.is_empty());
        assert!(!trie.contains("hello".chars()));
    }

    #[test]
    fn test_longest_match_iter() {
        let mut trie: Trie<char, ()> = Trie::new();
        trie.insert_seq("hello".chars());

        assert_eq!(trie.longest_match_iter("helloworld".chars(), 10), 5);
        assert_eq!(trie.longest_match_iter("world".chars(), 10), 0);
    }

    #[test]
    fn test_phoneme_trie() {
        let mut trie: Trie<&str, ()> = Trie::new();

        let hello_phonemes = ["h", "ə", "l", "oʊ"];
        let help_phonemes = ["h", "ɛ", "l", "p"];
        let world_phonemes = ["w", "ɜː", "l", "d"];

        trie.insert_seq(hello_phonemes.iter().copied());
        trie.insert_seq(help_phonemes.iter().copied());
        trie.insert_seq(world_phonemes.iter().copied());

        assert!(trie.contains(hello_phonemes.iter().copied()));
        assert!(trie.contains(help_phonemes.iter().copied()));
        assert!(!trie.contains(["h", "ə"].iter().copied())); // prefix only

        let test_phonemes = ["h", "ə", "l", "oʊ", "w", "ɜː", "l", "d"];
        assert_eq!(trie.longest_match(&test_phonemes, 10), 4); // matches "hello"
    }

    #[test]
    fn test_integer_trie() {
        let mut trie: Trie<u32, ()> = Trie::new();

        trie.insert_seq([1, 2, 3].iter().copied());
        trie.insert_seq([1, 2, 4].iter().copied());
        trie.insert_seq([5, 6, 7].iter().copied());

        assert!(trie.contains([1, 2, 3].iter().copied()));
        assert!(!trie.contains([1, 2].iter().copied()));

        let sequence = [1, 2, 3, 5, 6, 7];
        assert_eq!(trie.longest_match(&sequence, 10), 3);
    }

    // --- Counting trie tests ---

    #[test]
    fn test_count_trie_new_is_empty() {
        let trie: CountTrie<String> = CountTrie::new();
        assert_eq!(trie.get_count(std::iter::empty::<String>()), 0);
    }

    #[test]
    fn test_count_trie_increment_and_get_count() {
        let mut trie: CountTrie<String> = CountTrie::new();
        let seq = || ["the", "cat"].iter().map(|s| s.to_string());

        trie.increment(seq());
        assert_eq!(trie.get_count(seq()), 1);

        trie.increment(seq());
        assert_eq!(trie.get_count(seq()), 2);

        trie.increment(seq());
        assert_eq!(trie.get_count(seq()), 3);
    }

    #[test]
    fn test_count_trie_get_count_missing() {
        let trie: CountTrie<String> = CountTrie::new();
        assert_eq!(
            trie.get_count(["the", "cat"].iter().map(|s| s.to_string())),
            0
        );
    }

    #[test]
    fn test_count_trie_children_count_sum() {
        let mut trie: CountTrie<String> = CountTrie::new();

        // "the cat" x2, "the dog" x1
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "dog"].iter().map(|s| s.to_string()));

        // Children of "the" should sum to 3
        assert_eq!(
            trie.children_count_sum(std::iter::once("the".to_string())),
            3
        );

        // Children of root should sum to 0 (root's child "the" has no terminal,
        // because we never incremented just ["the"])
        assert_eq!(trie.children_count_sum(std::iter::empty::<String>()), 0);
    }

    #[test]
    fn test_count_trie_children_with_counts() {
        let mut trie: CountTrie<String> = CountTrie::new();

        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "dog"].iter().map(|s| s.to_string()));

        let mut children = trie.children_with_counts(std::iter::once("the".to_string()));
        children.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(children.len(), 2);
        assert_eq!(children[0], ("cat".to_string(), 2));
        assert_eq!(children[1], ("dog".to_string(), 1));
    }

    #[test]
    fn test_count_trie_children_missing_context() {
        let trie: CountTrie<String> = CountTrie::new();
        let children = trie.children_with_counts(std::iter::once("nonexistent".to_string()));
        assert!(children.is_empty());
    }

    #[test]
    fn test_count_trie_overlapping_prefixes() {
        let mut trie: CountTrie<String> = CountTrie::new();

        // Increment "a" (unigram) and "a b" (bigram) separately
        trie.increment(std::iter::once("a".to_string()));
        trie.increment(std::iter::once("a".to_string()));
        trie.increment(["a", "b"].iter().map(|s| s.to_string()));

        // "a" as a unigram has count 2
        assert_eq!(trie.get_count(std::iter::once("a".to_string())), 2);
        // "a b" as a bigram has count 1
        assert_eq!(trie.get_count(["a", "b"].iter().map(|s| s.to_string())), 1);
        // They are at different nodes, so counts are independent
    }

    #[test]
    fn test_count_trie_clear() {
        let mut trie: CountTrie<String> = CountTrie::new();
        trie.increment(std::iter::once("hello".to_string()));
        assert_eq!(trie.get_count(std::iter::once("hello".to_string())), 1);

        trie.clear();
        assert_eq!(trie.get_count(std::iter::once("hello".to_string())), 0);
    }

    #[test]
    fn test_count_trie_all_counts_empty() {
        let trie: CountTrie<String> = CountTrie::new();
        assert!(trie.all_counts().is_empty());
    }

    #[test]
    fn test_count_trie_all_counts() {
        let mut trie: CountTrie<String> = CountTrie::new();
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "dog"].iter().map(|s| s.to_string()));
        trie.increment(std::iter::once("a".to_string()));

        let mut counts = trie.all_counts();
        counts.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(counts.len(), 3);
        assert_eq!(counts[0], (vec!["a".to_string()], 1));
        assert_eq!(counts[1], (vec!["the".to_string(), "cat".to_string()], 2));
        assert_eq!(counts[2], (vec!["the".to_string(), "dog".to_string()], 1));
    }

    #[test]
    fn test_count_trie_all_counts_length_matches_len() {
        let mut trie: CountTrie<String> = CountTrie::new();
        trie.increment(["a", "b"].iter().map(|s| s.to_string()));
        trie.increment(["a", "c"].iter().map(|s| s.to_string()));
        trie.increment(std::iter::once("x".to_string()));

        assert_eq!(trie.all_counts().len(), trie.len());
    }

    #[test]
    fn test_count_trie_total_count_empty() {
        let trie: CountTrie<String> = CountTrie::new();
        assert_eq!(trie.total_count(), 0);
    }

    #[test]
    fn test_count_trie_total_count() {
        let mut trie: CountTrie<String> = CountTrie::new();
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "cat"].iter().map(|s| s.to_string()));
        trie.increment(["the", "dog"].iter().map(|s| s.to_string()));
        trie.increment(std::iter::once("a".to_string()));

        // 2 + 1 + 1 = 4
        assert_eq!(trie.total_count(), 4);
    }
}
