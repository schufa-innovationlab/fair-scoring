Getting Started
===============

Installation
------------

Install with ``pip`` directly:

.. sourcecode:: shell

   pip install fair-scoring


Introductory Example
--------------------
The following example shows how compute the equal opportunity bias.

Loading a Dataset
^^^^^^^^^^^^^^^^^
For this example, we use the compas dataset.

.. sourcecode:: python

   import pandas as pd

   # Load compas data
   dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
   df = pd.read_csv(dataURL)

   # Relevant data
   scores = df['decile_score']
   target = df['two_year_recid']
   attribute = df['race']

Any other dataset with continuous ``score`` and binary ``target`` and ``attribute`` will also work.

Computing the Bias
^^^^^^^^^^^^^^^^^^
We use the predefined equal opportunity bias for this example. This bias

.. sourcecode:: python

   from fairscoring.metrics import bias_metric_eo

   bias_eo = bias_metric_eo(scores, target, attribute, groups=['African-American', 'Caucasian'],favorable_target=0, prefer_high_scores=False)



Detailed Bias
^^^^^^^^^^^^^
To get a more detailed result, we can call the :code:`bias()` method:

.. sourcecode:: python

   result = bias_metric_eo.bias(scores, target, attribute, groups=['African-American', 'Caucasian'],favorable_target=0, n_permute=1000)

   print(f"Bias: {result.bias:.3f}")
   print(f"Pos: {100*result.pos_component:.0f}%")
   print(f"Neg: {100*result.neg_component:.0f}%")
   print(f"p-value: {result.p_value:.2f}")

.. note::
   The information of the result of a bias-computation depends on the metric and also on the call.
   In this case setting :code:`n_permute=1000` leads to a permutation test. Without this :code:`p_value` would be :code:`nan`.
