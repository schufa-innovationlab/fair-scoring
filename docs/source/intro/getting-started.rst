Getting Started
===============

Installation
------------

Install with `pip` directly:
.. sourcecode:: shell

   pip install git+https://github.com/schufa-innovationlab/fair-scoring.git

To install a specific version, use

.. sourcecode:: shell

   pip install git+https://github.com/schufa-innovationlab/fair-scoring.git@0.0.1

where `0.0.1` has to be replaced with any tag or branch.



Introductory Example
--------------------
The following example shows how compute the equal opportunity bias of the compas dataset

.. sourcecode:: python

   import pandas as pd
   from fairscoring.metrics import bias_eo

   # Load compas data
   dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
   df = pd.read_csv(dataURL)

   # Relevant data
   scores = 11 - df['decile_score']
   target = df['two_year_recid']
   attribute = df['race']

   # Compute the bias
   bias = bias_eo(scores, target, attribute, groups=['African-American', 'Caucasian'],favorable_target=0)

To get a more detailed result, we can call

.. sourcecode:: python

   result = bias_eo.bias(scores, target, attribute, groups=['African-American', 'Caucasian'],favorable_target=0, n_permute=1000)

   print(f"Bias: {result.bias:.3f}")
   print(f"Pos: {100*result.pos_component:.0f}%")
   print(f"Neg: {100*result.neg_component:.0f}%")
   print(f"p-value: {result.p_value:.2f}")

.. note::
   The information of the result of a bias-computation depends on the metric and also on the call.
   In this case setting :code:`n_permute=1000` leads to a permutation test, which results in :code:`p_value`.
