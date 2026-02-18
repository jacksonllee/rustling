Part-of-Speech Tagging
======================

Averaged Perceptron Tagger
---------------------------

The :py:class:`~rustling.tagging.AveragedPerceptronTagger` is a classic part-of-speech tagger
based on the averaged perceptron algorithm.

.. code-block:: python

   from rustling.tagging import AveragedPerceptronTagger

   # Initialize the tagger
   tagger = AveragedPerceptronTagger()

   # Fit on tagged sentences
   tagger.fit([
       [("The", "DT"), ("cat", "NN"), ("sat", "VBD")],
       [("A", "DT"), ("dog", "NN"), ("ran", "VBD")],
   ])

   # Predict tags for a new sentence
   result = tagger.predict(["The", "dog", "sat"])
   print(result)
   # ['DT', 'NN', 'VBD']
