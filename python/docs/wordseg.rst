Word Segmentation
=================

Longest String Matching
------------------------

The ``LongestStringMatching`` segmenter uses a greedy left-to-right longest match algorithm
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

The ``RandomSegmenter`` provides a random baseline for word segmentation.
No training is needed.

.. code-block:: python

   from rustling.wordseg import RandomSegmenter

   segmenter = RandomSegmenter(prob=0.3)
   result = segmenter.predict(["helloworld"])
   print(result)
   # e.g., [['hel', 'lo', 'wor', 'ld']] (varies due to randomness)
