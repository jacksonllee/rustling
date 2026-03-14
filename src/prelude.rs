//! Convenience re-exports of all `Base*` traits.
//!
//! ```rust
//! use rustling::prelude::*;
//! ```

pub use crate::chat::BaseChat;
pub use crate::chat::{BaseToken, BaseUtterance};
pub use crate::hmm::BaseHiddenMarkovModel;
pub use crate::lm::BaseLanguageModel;
pub use crate::ngram::BaseNgrams;
pub use crate::perceptron_pos_tagger::BaseTagger;
pub use crate::wordseg::{
    BaseHiddenMarkovModelSegmenter, BaseLongestStringMatching, BaseRandomSegmenter,
};
