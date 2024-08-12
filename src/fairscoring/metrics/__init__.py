"""
A collection fairness metrics to measure the bias of scores.

The package consists of multiple classes that cover different fairness concepts and distance metrics.
A concrete fairness metric is an instance of these classes with a concrete parametrization.


There are recommended **default instances** are already included:

=========================  =========================  =========================================
Metric                     Bias                       Class
=========================  =========================  =========================================
:const:`bias_metric_cal`   Calibration Bias           :class:`~.calibration.CalibrationMetric`
:const:`bias_metric_eo`    Equal Opportunity Bias     :class:`~.integral.WassersteinMetric`
:const:`bias_metric_pe`    Predictive Equality Bias   :class:`~.integral.WassersteinMetric`
=========================  =========================  =========================================

.. versionchanged:: 0.2.0
   The ``bias_`` prefixed constants are now prefixed by ``bias_metric_``

"""
from .calibration import CalibrationMetric
from .integral import WassersteinMetric

__all__ = ['bias_metric_cal', 'bias_metric_eo', 'bias_metric_pe']

bias_metric_cal = CalibrationMetric(name="Calibration", score_transform="quantile")
bias_metric_eo = WassersteinMetric(fairness_type="EO", name="Equal Opportunity", score_transform="quantile")
bias_metric_pe = WassersteinMetric(fairness_type="PE", name="Predictive Equality", score_transform="quantile")
