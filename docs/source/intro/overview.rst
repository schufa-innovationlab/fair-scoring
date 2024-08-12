Overview
========
The package contains fairness metrics for continuous risk scores.

Core Features
-------------
Analyzing Continuous Scores
    This packages focuses on the bias of continuous scores in contrast to other packages,
    which typically focus on the bias of decisions.

Multiple Algorithms and Variants
    This package contains a number of different algorithms for measuring the bias of continuous risk scores.
    Many of these algorithms can be parameterized.

Default Metrics
    You don't want to deal wih parameters and different algorithms?
    There are recommended default metrics ready for your use (see also :mod:`fairscoring.metrics`)

Model Agnostic
    The bias measures in this packages only require the pieces of information for a dataset:

    1. The continuous score
    2. The binary outcome or label
    3. The protected attribut or group information


Scientific Background
---------------------
The main implemented algorithms are described in the paper [BeDB24]_.
Experiments from this work can be found as jupyter notebooks :doc:`in the examples part <../examples/experiments>`.

Furthermore, the roc-based methods from [VoBC21]_ can also be used with this framework.

References
----------
.. [BeDB24] Becker, A.K. and Dumitrasc, O. and Broelemann, K.;
   Standardized Interpretable Fairness Measures for Continuous Risk Scores;
   Proceedings of the 41th International Conference on Machine Learning, 2024.


.. [VoBC21] Vogel, R., Bellet, A., Clémençon, S.; Learning Fair Scoring Functions: Bipartite Ranking under
   ROC-based Fairness Constraints; Proceedings of The 24th International Conference on Artificial
   Intelligence and Statistics, 2021.

