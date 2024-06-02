"""
A collection fairness metrics to measure the bias of scores.

The package consists of multiple classes that cover different fairness concepts and distance metrics.
A concrete fairness metric is an instance of these classes with a concrete parametrization.

There are recommended default instances are already included:

==================  =========================
Metric              Bias
==================  =========================
:const:`bias_cal`   Calibration Bias
:const:`bias_ind`   Independence Bias
:const:`bias_eo`    Equal Opportunity Bias
:const:`bias_pe`    Predictive Equality Bias
==================  =========================

"""
from ._calibration import CalibrationMetric
from ._integral import WassersteinMetric

bias_cal = CalibrationMetric(name="Calibration", score_transform="quantile")
bias_ind = WassersteinMetric(fairness_type="IND", name="Independence", score_transform="quantile")
bias_eo = WassersteinMetric(fairness_type="EO", name="Equal Opportunity", score_transform="quantile")
bias_pe = WassersteinMetric(fairness_type="PE", name="Predictive Equality", score_transform="quantile")
