"""
Intergal bias metrics that measure the differences between cumulative distribution functions.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import Literal, Optional

from .base import TwoGroupBiasResult, TwoGroupMetric
from fairscoring.utils import _ENCODING_FAVORABLE_OUTCOME, _ENCODING_UNFAVORABLE_OUTCOME
from abc import abstractmethod

INDEPENDENCE = 'IND'
EQUAL_OPPORTUNITY = 'EO'
PREDICTIVE_EQUALITY = 'PE'

__all__ = ['IntegralMetric', 'WassersteinMetric', 'INDEPENDENCE', 'EQUAL_OPPORTUNITY', 'PREDICTIVE_EQUALITY']


class IntegralMetric(TwoGroupMetric):
    """
    Base Class for Integral Metrics that compare cdfs.

    Parameters
    ----------
    fairness_type: {"IND", "EO", "PE"}
        Specifies the type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality),

    name: str
        Name of the Metric

    score_transform: {"rescale","quantile",None}
        A transformation of the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`.
          In this case, the :meth:`~fairscoring.metrics._integral.IntegralMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.
    """
    def __init__(self, fairness_type:Literal["IND", "EO", "PE"], name:str, score_transform: Optional[Literal["rescale", "quantile"]] = None):
        super().__init__(name=name, score_transform=score_transform)
        self.fairness_type = fairness_type

    def _split_groups(self, scores: ArrayLike, target: ArrayLike, attribute: np.ndarray, groups: list,
                      return_total: bool = False) -> list[np.ndarray]:
        """
        Split the data into groups.

        Parameters
        ----------
        scores: ArrayLike
            A list of scores

        target: ArrayLike
            The binary target values. Must have the same length as `scores`.

        attribute: ndarray
            The protected attribute. Must have the same length as `scores`.

        groups: list
            A list of groups. Each group is given by a value of the protected attribute.
            A value of `None` is used to define a group with all elements that are not in another group.

        return_total: bool, default = False
            If set to true, an additional group with the data of the whole population will be returned.

        Returns
        -------
        list of ndarrays
            Returns for each group an array binary filter, that is 'True' for those elements that belong to the group.

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks are performed.

        Child classes might override or extend this method to influence which samples are used,
        e.g. to restrict it to one class for the separation bias.
        """
        # Split groups
        groups = super()._split_groups(scores, target, attribute, groups, return_total)

        # Filter for separation metrics
        if self.fairness_type == EQUAL_OPPORTUNITY:
            filter = (target == _ENCODING_FAVORABLE_OUTCOME)
            groups = [np.logical_and(grp, filter) for grp in groups]
        elif self.fairness_type == PREDICTIVE_EQUALITY:
            filter = (target == _ENCODING_UNFAVORABLE_OUTCOME)
            groups = [np.logical_and(grp, filter) for grp in groups]

        return groups

    def _groupwise_data(self, scores: ArrayLike, target: ArrayLike, attribute: ArrayLike,
                        groups_filter: list[np.ndarray]) -> list[ArrayLike]:
        """
        Gets groupwise data. For integraL metrics, this is an array of scores for each group.

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
        data: list of scores
            One array of scores for each group

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks need to be performed.
        """
        return [scores[grp] for grp in groups_filter]

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

        min_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.

        Returns
        -------
        bias: TwoGroupBiasResult
            The stored results of the bias computation
        """
        if total is None:
            x, cdf_dis, cdf_adv = _cdf_from_scores(dis_grp, adv_grp)
        else:
            x, cdf_dis, cdf_adv, total = _cdf_from_scores(dis_grp, adv_grp, total)

        return self._compute_integral_bias(x, cdf_dis, cdf_adv)

    @abstractmethod
    def _compute_integral_bias(self, x:ArrayLike, cdf_dis:ArrayLike, cdf_adv:ArrayLike) -> IntegralBiasResult:
        """
        Computes the bias based on the cdfs

        Parameters
        ----------
        x: ArrayLike
            X-values at with the cdf is evaluated

        cdf_dis: ArrayLike
            Array with the cdf-values of the disadvantaged corresponding to the x-values.

        cdf_adv:
            Array with the cdf-values of the advantaged corresponding to the x-values.

        Returns
        -------
        bias: IntegralBiasResult
            The stored results of the bias computation
        """
        pass


class IntegralBiasResult(TwoGroupBiasResult):
    """
    An extended bias result that also stores groupwise cumulative distribution functions (cdfs).


    Parameters
    ----------
    bias: float
        The bias value

    pos: float
        The positive component of the bias

    neg: float
        The negative component of the bias

    cdf_x: ArrayLike
        x-values at which the cdfs are stored. This array is 1-dimensional

    cdfs: List of ArrayLike
        A list of cdfs.

    Attributes
    ----------
    bias: float
        The bias value

    pos: float
        The positive component of the bias

    neg: float
        The negative component of the bias

    cdf_x: ArrayLike
        x-values at which the cdfs are stored. This array is 1-dimensional

    cdfs: List of ArrayLike
        A list of cdfs.
    """
    def __init__(self, bias: float, pos: float, neg: float, cdf_x: ArrayLike, cdfs: ArrayLike):
        super().__init__(bias=bias, pos=pos, neg=neg)
        self.cdf_x = cdf_x
        self.cdfs = cdfs


class WassersteinMetric(IntegralMetric):
    """
    A metric that measures the differences between distributions using the Wasserstein Distance [BeDB24]_.

    This metric can be used to measure independence and separation bias.
    The `fairness_type`-parameter specifies which bias to measure and hence which distribution will be compared.

    Parameters
    ----------
    fairness_type: {"IND", "EO", "PE"}
        Specifies the type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality),

    name: str
        Name of the Metric

    score_transform: {"rescale","quantile",None}
        A transformation of the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`.
          In this case, the :meth:`~fairscoring.metrics._integral.WassersteinMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.

    p: float, default=1
        Exponent for the Wasserstein Distance.
        Use the default of 1 to get the Earthmover Distance
    """
    def __init__(self, fairness_type: Literal["IND", "EO", "PE"], name: str, score_transform: Optional[Literal["rescale", "quantile"]] = None, p: float = 1):
        super().__init__(fairness_type=fairness_type, name=name, score_transform=score_transform)
        self.p = p

    def _compute_integral_bias(self, x: ArrayLike, cdf_dis: ArrayLike, cdf_adv: ArrayLike) -> IntegralBiasResult:
        """
        Computes the wasserstein distance between two cdfs

        Parameters
        ----------
        x: ArrayLike
            X-values at with the cdf is evaluated

        cdf_dis: ArrayLike
            Array with the cdf-values of the disadvantaged corresponding to the x-values.

        cdf_adv:
            Array with the cdf-values of the advantaged corresponding to the x-values.

        Returns
        -------
        bias: TwoGroupBiasResult
            The stored results of the bias computation
        """
        # Compute the differences between pairs of successive values of u and v.
        # TODO: maybe use min/max score
        deltas = np.diff(x)

        # determine indices of positive and negative parts
        idx_pos = np.where((1 - cdf_dis[:-1]) - (1 - cdf_adv[:-1]) >= 0)
        idx_neg = np.where((1 - cdf_dis[:-1]) - (1 - cdf_adv[:-1]) < 0)

        bias, pos_bias, neg_bias, p_val = _wasserstein_integral(deltas, idx_neg, idx_pos, self.p, cdf_dis, cdf_adv)

        return IntegralBiasResult(bias, pos_bias, neg_bias, cdf_x=x, cdfs=[cdf_dis, cdf_adv])


def _wasserstein_integral(deltas, idx_neg, idx_pos, p, u_cdf, v_cdf):
    """
    TODO Documentation

    Parameters
    ----------
    deltas
    idx_neg
    idx_pos
    p
    u_cdf
    v_cdf

    Returns
    -------

    """
    p_val = np.nan
    if p == 1:
        positive_component = np.sum(np.multiply(np.abs(u_cdf[idx_pos] - v_cdf[idx_pos]), deltas[idx_pos]))
        negative_component = -1 * np.sum(np.multiply(np.abs(u_cdf[idx_neg] - v_cdf[idx_neg]), deltas[idx_neg]))
        net_distance = np.sum(np.abs(positive_component) + np.abs(negative_component))
    elif p == 2:
        positive_component = np.sqrt(np.sum(np.multiply(np.square(u_cdf[idx_pos] - v_cdf[idx_pos]), deltas[idx_pos])))
        negative_component = -1 * np.sqrt(
            np.sum(np.multiply(np.square(u_cdf[idx_neg] - v_cdf[idx_neg]), deltas[idx_neg])))
        net_distance = np.sqrt(np.sum(np.square(positive_component) + np.square(negative_component)))
    else:
        positive_component = np.power(
            np.sum(np.multiply(np.power(np.abs(u_cdf[idx_pos] - v_cdf[idx_pos]), p), deltas[idx_pos])), 1 / p)
        negative_component = -1 * np.power(
            np.sum(np.multiply(np.power(np.abs(u_cdf[idx_neg] - v_cdf[idx_neg]), p), deltas[idx_neg])), 1 / p)
        net_distance = np.power(
            np.sum(np.power(np.abs(positive_component), p) + np.power(np.abs(negative_component), p)), 1 / p)
    return net_distance, positive_component, negative_component, p_val


def _cdf_from_scores(*args:ArrayLike):
    """
    Computes the cdfs of different univariate samples.

    All cdfs are evaluated at the same point. Hence, the function will only produce a single array auf x-values.

    Parameters
    ----------
    *args: ArrayLike
        One arrays of scores for each group.

    Returns
    -------
    x: ArrayLike
        X-values at with the cdf is evaluated

    *cdfs: ArrayLike
        For each group there is an array with the cdf-values corresponding to the x-values.

    """
    # calculate functions and integral
    # sort
    s_sorter = [np.argsort(s) for s in args]

    # concatenate to get all x-values, for which the cdfs are computed
    x = np.concatenate(args)

    # x = np.unique(x)  # TODO: preferred, but there seems to be bug with `.searchsorted(x[:-1], 'right')`
    x.sort(kind='mergesort')

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    cdfs = []
    for s_val, s_sort in zip(args, s_sorter):
        cdf_indices = s_val[s_sort].searchsorted(x, 'right')
        cdf = cdf_indices / s_val.size
        cdfs.append(cdf)

    return x, *cdfs
