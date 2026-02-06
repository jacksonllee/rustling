Part-of-Speech Tagging
======================

Averaged Perceptron Tagger
---------------------------

The ``AveragedPerceptronTagger`` is a fast and accurate part-of-speech tagger
based on the averaged perceptron algorithm.

.. code-block:: python

   from rustling.taggers import AveragedPerceptronTagger

   # Initialize the tagger
   tagger = AveragedPerceptronTagger()

   # Train on tagged sentences
   tagger.fit([
       [("The", "DT"), ("cat", "NN"), ("sat", "VBD")],
       [("A", "DT"), ("dog", "NN"), ("ran", "VBD")],
   ])

   # Predict tags for new sentences
   result = tagger.predict([["The", "dog", "sat"]])
   print(result)
   # [[('The', 'DT'), ('dog', 'NN'), ('sat', 'VBD')]]
