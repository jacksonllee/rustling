Word Segmentation
=================

Hidden Markov Model
-------------------

The :py:class:`~rustling.wordseg.HiddenMarkovModelSegmenter` uses a supervised hidden Markov model
with BMES tagging and Viterbi decoding to segment unsegmented text into words.

.. code-block:: python

   from rustling.wordseg import HiddenMarkovModelSegmenter

   model = HiddenMarkovModelSegmenter()
   model.fit([
       ("this", "is", "a", "sentence"),
       ("that", "is", "not", "a", "sentence"),
   ])
   result = model.predict(["thatisadog", "thisisnotacat"])
   print(result)
   # [['that', 'is', 'a', 'd', 'o', 'g'], ['this', 'is', 'not', 'a', 'c', 'a', 't']]

DAG-HMM Segmenter
------------------

The :py:class:`~rustling.wordseg.DAGHMMSegmenter` is a jieba-style hybrid segmenter that combines
dictionary-based DAG (directed acyclic graph) segmentation with an HMM fallback for
out-of-vocabulary spans.

.. code-block:: python

   from rustling.wordseg import DAGHMMSegmenter

   model = DAGHMMSegmenter()
   model.fit_segmented([
       ("this", "is", "a", "sentence"),
       ("that", "is", "not", "a", "sentence"),
   ])
   result = model.predict(["thatisadog", "thisisnotacat"])
   print(result)

Longest String Matching
------------------------

The :py:class:`~rustling.wordseg.LongestStringMatching` segmenter uses a greedy left-to-right longest match algorithm
to segment unsegmented text into words.

.. code-block:: python

   from rustling.wordseg import LongestStringMatching

   model = LongestStringMatching(max_word_length=4)
   model.fit([
       ("this", "is", "a", "sentence"),
       ("that", "is", "not", "a", "sentence"),
   ])
   result = model.predict(["thatisadog", "thisisnotacat"])
   print(result)
   # [['that', 'is', 'a', 'd', 'o', 'g'], ['this', 'is', 'not', 'a', 'c', 'a', 't']]

Random Segmenter
----------------

The :py:class:`~rustling.wordseg.RandomSegmenter` provides a random baseline for word segmentation.
No training is needed.

.. code-block:: python

   from rustling.wordseg import RandomSegmenter

   segmenter = RandomSegmenter(prob=0.3)
   result = segmenter.predict(["helloworld"])
   print(result)
   # e.g., [['hel', 'lo', 'wor', 'ld']] (varies due to randomness)
