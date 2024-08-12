Concepts
========
The aim of this package is to provide different **metrics** that measure bias in continuous scoring systems.

Fairness Concepts
-----------------
Given a scores or prediction :math:`S\in\mathbb{R}`, a binary target variable :math:`Y\in\{0,1\}` and a (binary) group
:math:`A\in\{a,b\}`, we can formulate the following concepts for group fairness:

Independence Bias
    A score fulfills independence fairness, if the score distribution is independent of the group, i.e. :math:`S\perp \!\!\! \perp A`.
    **Note** that a (hypothetical) perfect predictor (where prediction are equal to the outcome) will fail independence fairness
    if the basic risk rates differ between groups. For this reason, we do not recommend to use it, if a reliable target variable
    is available.

    To quantify the bias, one can measure the difference between the two distributions :math:`S|A=a` and :math:`S|A=b`.
    In our package, this is done via :class:`~fairscoring.metrics.integral.WassersteinMetric` (with ``fairness_type='IND'``),
    which computes the Wasserstein distance between these two distributions.

Separation Bias
    A score fulfills separation fairness, if the score distribution is independent of the group given the outcome,
    i.e. :math:`S\perp \!\!\! \perp A \,|\, Y`.

    We implement two metrics to measure the magnitude of the separation bias:

    1. Equal Opportunity compares the score distributions of samples with favorable outcome,
       i.e. :math:`S|A=a,Y=0` and :math:`S|A=b,Y=0`.
    2. Predictive Equality compares the score distributions of samples with unfavorable outcome,
       i.e. :math:`S|A=a,Y=1` and :math:`S|A=b,Y=1`.

    Both metrics are implemented in :class:`~fairscoring.metrics.integral.WassersteinMetric` (either with ``fairness_type='EO'``
    or with ``fairness_type='PE'``). These classes compute the Wasserstein distance between above mentioned distributions.

Sufficiency / Calibration Bias
    A score fulfills calibration fairness, if the outcome is independent of the group given the score,
    i.e. :math:`Y\perp \!\!\! \perp A \,|\, S`. For scores, we use calibration and sufficiency bias synonymously.
    Note that for fairness of (binary) decisions, the term sufficiency is used. In this sence, sufficiency is the broader concept.

    In this package, the magnitude of the calibration bias can be measured via
    :class:`~fairscoring.metrics.calibration.CalibrationMetric`, which measures the differences between calibration curves.


Further Metrics
^^^^^^^^^^^^^^^
Beside the metrics, there are other ways to measure differences between groups.
The downside of these metrics is the lack of a clear connection to one of the fairness concepts.
Nevertheless, this packages also provides some of these metrics:

ROC / ABROCA Metrics
    These metrics measure the *Absolute Between-ROC Area* (ABROCA). ROC-Base measures are available through the
    :class:`~fairscoring.metrics.roc.ROCBiasMetric`.

    We distinguish the following to metrics that compare different roc-curves:

    1. Setting ``bias_type='roc'`` computes the area between the groupwise roc-curves.
    2. Setting ``bias_type='xroc'`` (cross-roc) builds roc-curves with :math:`Y=0` samples from one group and :math:`Y=1`
       samples from the other group.

API Concepts
------------
Metrics
^^^^^^^
In this package, each bias metric is implemented as an instance of the :class:`~fairscoring.metrics.base.BaseBiasMetric` class.
The main method of this class is :meth:`~fairscoring.metrics.base.BaseBiasMetric.bias`, which takes three arrays and
some metadata to compute the bias. These three arrays are:

1. The score value of each sample
2. The target variable / the actual outcome
3. The attribute or group each sample belongs to

As a convenience function, :class:`~fairscoring.metrics.base.BaseBiasMetric` is also callable. Calling a metric will
return the bias as a single float value.

Bias Results
^^^^^^^^^^^^
The method :meth:`~fairscoring.metrics.base.BaseBiasMetric.bias` returns a :class:`~fairscoring.metrics.base.BiasResult`
object. In its basic form, this class only contains a single ``bias`` value. The idea is to extend this class to return
further data specific to some bias metrics.
Most notable is the :class:`~fairscoring.metrics.base.TwoGroupBiasResult` that is currently supported by each bias metric.
Beside the pure bias value, it also contains a split into positive and negative bias.

Plots
^^^^^
The package contains a number of plots that visualize the bias.
These allow for a better understanding of the bias. Each plot takes an axes, which allows to combine multiple
plots into one bias figure.

See the :mod:`fairscoring.plots` module for a list of supported plots.
Examples for their usage can be found in the :doc:`examples <../examples/examples>` section.