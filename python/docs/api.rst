API Reference
=============

Word Segmentation
-----------------

.. py:module:: rustling.wordseg

.. py:class:: LongestStringMatching(max_word_length: int)

   Greedy left-to-right longest match segmenter.

   :param max_word_length: Maximum word length to consider during segmentation.

   .. py:method:: fit(sentences: list[tuple[str, ...]]) -> None

      Train the segmenter on a list of segmented sentences.

      :param sentences: A list of tuples, where each tuple contains the words of a sentence.

   .. py:method:: predict(unsegmented: list[str]) -> list[list[str]]

      Segment a list of unsegmented strings.

      :param unsegmented: A list of unsegmented strings.
      :returns: A list of segmented sentences, where each sentence is a list of words.

.. py:class:: RandomSegmenter(prob: float)

   Random baseline segmenter.

   :param prob: Probability of inserting a word boundary at each character position.

   .. py:method:: predict(unsegmented: list[str]) -> list[list[str]]

      Randomly segment a list of unsegmented strings.

      :param unsegmented: A list of unsegmented strings.
      :returns: A list of segmented sentences, where each sentence is a list of words.

Part-of-Speech Tagging
----------------------

.. py:module:: rustling.taggers

.. py:class:: AveragedPerceptronTagger()

   Averaged perceptron part-of-speech tagger.

   .. py:method:: fit(sentences: list[list[tuple[str, str]]]) -> None

      Train the tagger on a list of tagged sentences.

      :param sentences: A list of sentences, where each sentence is a list of (word, tag) tuples.

   .. py:method:: predict(sentences: list[list[str]]) -> list[list[tuple[str, str]]]

      Predict tags for a list of sentences.

      :param sentences: A list of sentences, where each sentence is a list of words.
      :returns: A list of tagged sentences, where each sentence is a list of (word, tag) tuples.
