"""
ROC-Base metrics.
"""
import numpy as np
from numpy._typing import ArrayLike
from sklearn.metrics import roc_curve

from ._base import BaseBiasMetric, TwoGroupMixin, TwoGroupBiasResult
from typing import Literal, Tuple


class ROCBiasMetric(TwoGroupMixin, BaseBiasMetric):
    """
    ROC-Based Fairness Metrics.

    These metrics compute the absolute between ROC area (ABROCA) of two roc-curve.
    The following two metrics can be distingushed:

    `roc`
        Compares the classic roc-curves of two groups with each other.

    `xroc`
        Builds roc curves with class 0 samples from one group and class 1 samples from the other group.

    Parameters
    ----------
    name: str
        Name of the metric.

    bias_type: {"roc", "xroc"}

    Notes
    -----
    This metric is not influenced by monotonic score transformations.
    For this reason, no `score_transform` parameter is provided.

    """
    def __init__(self, name:str, bias_type:Literal["roc","xroc"]):
        super().__init__(name=name, score_transform=None)
        self.bias_type = bias_type

    def _groupwise_data(self, scores: ArrayLike, target: ArrayLike, attribute: ArrayLike,
                        groups_filter: list[np.ndarray]) -> list:
        """
        Gets groupwise data. For roc-based metrics, this is a pair `(scores_0,scores_1)` for each group.
        `scores_0` are here the scores of target class `0` (favourable outcome) and `scores_1` are the
        samples with the unfavourable outcome.

        Parameters
        ----------
        scores: ArrayLike
            A list of scores

        target: ArrayLike
            The binary target values. Must have the same length as `scores`.

        attribute: ndarray
            The protected attribute. Must have the same length as `scores`.

        groups_filter: list of ndarray
            A list of group filters. Each is a boolean vector that is True for elements belonging to the group.

        Returns
        -------
        data: list of (score, target)
            One pair of `(score, target)` arrays for each group

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks need to be performed.
        """
        data = [(scores[flt & (target==0)],scores[flt & (target==1)]) for flt in groups_filter]
        return data

    def _compute_bias(self, dis_grp, adv_grp, total=None, min_score=None, max_score=None) -> TwoGroupBiasResult:
        """
        Computes the integral bias

        Parameters
        ----------
        dis_grp: ArrayLike
            Scores of the disadvantaged group.

        adv_grp: ArrayLike
            Scores of the advantaged group.

        total: ArrayLike, Optional
            Scores of the total population

        min_score: float
            The minimal score. This might influence the bias computation, e.g. by defining the integral bounds.
            Ignored for roc-metrics.

        min_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.
            Ignored for roc-metrics.

        Returns
        -------
        bias: TwoGroupBiasResult
            The stored results of the bias computation

        Raises
        ------
        ValueError
            If an invalid `bias_type`is provided.
        """
        if self.bias_type == "roc":
            bias, pos, neg = _abroca(dis_grp[0], dis_grp[1], adv_grp[0], adv_grp[1])
        elif self.bias_type == "xroc":
            bias, pos, neg = _abroca(dis_grp[0], adv_grp[1], adv_grp[0], dis_grp[1])
        else:
            raise ValueError("Invalid bias_type '" + self.bias_type + "'. Only 'roc' and 'xroc' are supported")

        return TwoGroupBiasResult(bias, pos=pos, neg=neg)


def _abroca(a0:ArrayLike, a1:ArrayLike, b0:ArrayLike, b1:ArrayLike) -> Tuple[float, float, float]:
    """
    Compute the absolute between roc area (ABROCA).

    For this method, two roc curves (a and b) are computed.
    Each roc curve is based on class 0 and class 1 samples.

    Parameters
    ----------
    a0: ArrayLike
        Scores of class 0 for roc-curve a

    a1: ArrayLike
        Scores of class 1 for roc-curve a

    b0: ArrayLike
        Scores of class 0 for roc-curve b

    b1: ArrayLike
        Scores of class 1 for roc-curve b

    Returns
    -------
    bias: float
        The absolute bias

    bias_pos: float
        The positive component of the bias

    bias_neg: float
        The negative component of the bias
    """
    fpr_a, tpr_a = _roc(a0, a1)
    fpr_b, tpr_b = _roc(b0, b1)

    # merge and sort all fpr (x) values
    all_fpr = np.concatenate((fpr_a, fpr_b))
    all_fpr.sort(kind='mergesort')

    # interpolation of y (linear)
    all_tpr_a = np.interp(all_fpr, fpr_a, tpr_a)
    all_tpr_b = np.interp(all_fpr, fpr_b, tpr_b)

    # differences and positive/negative parts
    differences = np.array([a_i - b_i for a_i, b_i in zip(all_tpr_a, all_tpr_b)])
    differences_pos = differences.copy()
    differences_pos[differences < 0] = 0
    differences_neg = differences.copy()
    differences_neg[differences > 0] = 0

    # integrate
    bias_pos = np.trapz(np.abs(differences_pos), x=all_fpr)
    bias_neg = np.trapz(np.abs(differences_neg), x=all_fpr)

    # total bias
    bias = np.abs(bias_pos + bias_neg)

    return bias, bias_pos, bias_neg


def _roc(s0:ArrayLike, s1:ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes the roc curve from two sets of scores, one for class 0 and one for class 1

    Parameters
    ----------
    s0: ArrayLike
        Scores of class 0

    s1: ArrayLike
        Scores of class 1

    Returns
    -------
    fpr: ArrayLike
        False Positive Values of the roc-curve (x-axis)

    tpr: ArrayLike
        True Positive Values of the roc-curve (y-axis)

    Notes
    -----
    The true positive rate is related to class 0 while the false positive rate is related to class 1.
    """
    s = np.concatenate([s0,s1])
    y = np.concatenate([np.zeros(s0.shape), np.ones(s1.shape)])
    fpr, tpr, _ = roc_curve(y, s, pos_label=0)

    return fpr, tpr

# Default instances
bias_roc = ROCBiasMetric("ROC bias", bias_type="roc")
bias_xroc = ROCBiasMetric("xROC bias", bias_type="xroc")


