"""Shared HKCanCor data loader for all benchmarks.

All benchmarks use the HKCanCor (Hong Kong Cantonese Corpus) via pycantonese
as a unified data source. This module loads the corpus and converts it to the
formats needed by each benchmark category (tagging, wordseg, lm).
"""

from __future__ import annotations


def load_hkcancor(min_sent_length: int = 3) -> list[list[tuple[str, str]]]:
    """Load tagged sentences from HKCanCor via pycantonese.

    Parameters
    ----------
    min_sent_length : int, default=3
        Minimum number of tokens per sentence to include.

    Returns
    -------
    list[list[tuple[str, str]]]
        Tagged sentences: [[(word, tag), ...], ...]
    """
    try:
        import pycantonese
    except ImportError:
        raise ImportError(
            "pycantonese is required for loading HKCanCor data. "
            "Install with: pip install pycantonese"
        )

    corpus = pycantonese.hkcancor()
    tagged_sents = []
    for sent_tokens in corpus.tokens(by_utterances=True):
        sent = [(token.word, token.pos) for token in sent_tokens if token.pos]
        if len(sent) >= min_sent_length:
            tagged_sents.append(sent)

    return tagged_sents


def tagging_data(
    tagged_sents: list[list[tuple[str, str]]],
    train_ratio: float = 0.8,
) -> tuple[list[list[tuple[str, str]]], list[list[str]]]:
    """Convert HKCanCor data to POS tagging format.

    Parameters
    ----------
    tagged_sents : list[list[tuple[str, str]]]
        Tagged sentences from load_hkcancor().
    train_ratio : float, default=0.8
        Fraction of data to use for training.

    Returns
    -------
    tuple[list[list[tuple[str, str]]], list[list[str]]]
        (training_data, test_sentences)
        training_data: tagged sentences [(word, tag), ...]
        test_sentences: untagged sentences [word, ...]
    """
    split_idx = int(len(tagged_sents) * train_ratio)
    training_data = tagged_sents[:split_idx]
    test_sentences = [[word for word, tag in sent] for sent in tagged_sents[split_idx:]]
    return training_data, test_sentences


def wordseg_data(
    tagged_sents: list[list[tuple[str, str]]],
    train_ratio: float = 0.8,
) -> tuple[list[tuple[str, ...]], list[str]]:
    """Convert HKCanCor data to word segmentation format.

    Parameters
    ----------
    tagged_sents : list[list[tuple[str, str]]]
        Tagged sentences from load_hkcancor().
    train_ratio : float, default=0.8
        Fraction of data to use for training.

    Returns
    -------
    tuple[list[tuple[str, ...]], list[str]]
        (training_data, test_sentences)
        training_data: segmented sentences as word tuples
        test_sentences: unsegmented concatenated strings
    """
    split_idx = int(len(tagged_sents) * train_ratio)
    training_data = [
        tuple(word for word, tag in sent) for sent in tagged_sents[:split_idx]
    ]
    test_sentences = [
        "".join(word for word, tag in sent) for sent in tagged_sents[split_idx:]
    ]
    return training_data, test_sentences


def lm_data(
    tagged_sents: list[list[tuple[str, str]]],
    train_ratio: float = 0.8,
) -> list[list[str]]:
    """Convert HKCanCor data to language model format.

    Parameters
    ----------
    tagged_sents : list[list[tuple[str, str]]]
        Tagged sentences from load_hkcancor().
    train_ratio : float, default=0.8
        Fraction of data to use for training.

    Returns
    -------
    list[list[str]]
        Sentences of words (just the word sequences, no tags).
    """
    split_idx = int(len(tagged_sents) * train_ratio)
    return [[word for word, tag in sent] for sent in tagged_sents[:split_idx]]
