Part-of-Speech Tagging
======================

Averaged Perceptron Tagger
---------------------------

The :py:class:`~rustling.perceptron_pos_tagger.AveragedPerceptron` is a classic part-of-speech tagger
based on the averaged perceptron algorithm.

.. code-block:: python

   from rustling.perceptron_pos_tagger import AveragedPerceptron

   # Initialize the tagger
   tagger = AveragedPerceptron()

   # Fit on sequences and their tags
   tagger.fit(
       [["The", "cat", "sat"], ["A", "dog", "ran"]],
       [["DT", "NN", "VBD"], ["DT", "NN", "VBD"]],
   )

   # Predict tags for new sentences
   result = tagger.predict([["The", "dog", "sat"]])
   print(result)
   # [['DT', 'NN', 'VBD']]


