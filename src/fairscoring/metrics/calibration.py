"""
A module to define the calibration bias metric.
"""
import numpy as np
from typing import Union, Iterable, Any, Tuple, Literal, Optional
from numpy.typing import ArrayLike
from .base import TwoGroupMetric, TwoGroupBiasResult
from fairscoring.utils import _ENCODING_FAVORABLE_OUTCOME

__all__ = ['CalibrationMetric']


def get_bins(y_pred,
             n_bins: int = 100,
             ranking: bool = True,
             rescale: bool = False) -> Union[int, float, complex, np.ndarray]:
    """
    Get bins for the calibration curve

    Parameters
    ----------
    y_pred: array of float
        scores / predictions
    n_bins: integer
        number of bins
    ranking: boolean
        if True, quantile-based bins are used
    rescale: boolean
        if True, values are min-max-rescaled

    Returns
    -------

    """
    if ranking:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_pred, quantiles * 100)
        if rescale:
            bins[0] = 0
            bins[-1] = 1
    else:
        if rescale:
            bins = np.linspace(0.0, 1, n_bins + 1)
        else:
            bins = np.linspace(min(y_pred), max(y_pred), n_bins + 1)
    return bins


def get_calibration_curve(y_true: Union[np.ndarray, Iterable, int, float],
                          y_pred: Union[np.ndarray, Iterable, int, float],
                          bins: Union[np.ndarray, Iterable, int, float]) \
        -> Any:
    """
    determine x and y values of the calibration curve for given bins

    Parameters
    ----------
    y_true: array-like
        true y values (i.e. CLASS)

    y_pred: array-like
        predicted scores

    bins: array-like
        bin cutpoints

    Returns
    -------
    mean predicted values (x-values)

    """
    # TODO: Rescaling not implemented
    # code from sklearn.calibration.calibration_curve:
    binids = np.searchsorted(bins[1:], y_pred)

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins) - 1)
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins) - 1)
    bin_total = np.bincount(binids, minlength=len(bins) - 1)

    nonzero = bin_total != 0
    fraction_of_positives = np.where(nonzero, bin_true / bin_total, np.nan)
    mean_predicted_value = np.where(nonzero, bin_sums / bin_total, np.nan)

    # add first and last value
    # mean_predicted_value = list(mean_predicted_value)
    # fraction_of_positives = list(fraction_of_positives)
    # mean_predicted_value = [0] + mean_predicted_value + [1]
    # fraction_of_positives = [0] + fraction_of_positives + [1]

    return mean_predicted_value[:(len(bins))], fraction_of_positives[:(len(bins))]


class CalibrationMetric(TwoGroupMetric):
    """
    This metric measures the differences between two calibration curves [BeDB24]_.

    Calibration is a way to measure sufficiency bias for continuous scores.
    The `weighting` parameter specifies how differences over the total score range are weighted.

    This metric returns a :class:`~fairscoring.metrics.base.TwoGroupBiasResult` object, which allows to split
    the bias in positive and negative parts.

    Parameters
    ----------
    weighting: ("quantiles", "scores")
        Integral over quantiles / scores

    n_bins: int
        Number of bins that are used to compute the calibration curves

    name: str, default="Calibration"
        Name of metric.

    score_transform: {"rescale","quantile",None}
        A transformation of the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`.
          In this case, the :meth:`~fairscoring.metrics._calibration.CalibrationMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.
    """

    def __init__(self, weighting:str = "quantiles", n_bins:int=50, name:str= "Calibration",
                 score_transform: Optional[Literal["rescale", "quantile"]] = None):
        super().__init__(name=name, score_transform=score_transform)
        self.weighting = weighting
        self.n_bins = n_bins

    def _groupwise_data(self, scores: ArrayLike, target: ArrayLike, attribute: ArrayLike, groups_filter: list[np.ndarray]
    ) -> list[Tuple[ArrayLike, ArrayLike]]:
        """
        Gets groupwise data. For calibration metrics, this is a pair `(scores,target)` for each group.

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
        return [(scores[grp], target[grp]) for grp in groups_filter]

    def _compute_bias(self, dis_grp, adv_grp, total=None, min_score=None, max_score=None) -> TwoGroupBiasResult:
        """
        Computes the calibration bias

        Parameters
        ----------
        dis_grp: tuple of (ArrayLike, ArrayLike)
            `(score, target)` pair for the disadvantaged group.

        adv_grp: list of (ArrayLike, ArrayLike)
            `(score, target)` pair for the advantaged group.

        total: list of (ArrayLike, ArrayLike), Optional
            `(score, target)` pair for the total population

        min_score: float, optional
            Minimal score. This is only used when `score_transform` is set to `"rescale"`.

        max_score: float, optional
            Maximal score. This is only used when `score_transform` is set to `"rescale"`.

        Returns
        -------
        bias: TwoGroupBiasResult
            The stored results of the bias computation
        """
        # Unpack parameters
        y_true_dis = dis_grp[1] == _ENCODING_FAVORABLE_OUTCOME
        y_pred_dis = dis_grp[0]

        y_true_ad = adv_grp[1] == _ENCODING_FAVORABLE_OUTCOME
        y_pred_ad = adv_grp[0]

        if total is None:
            y_true = np.concatenate([y_true_dis, y_true_ad])
            y_pred = np.concatenate([y_pred_dis, y_pred_ad])
        else:
            y_true = total[1] == _ENCODING_FAVORABLE_OUTCOME
            y_pred = total[0]

        # Compute the bins equidistance is in the (transformed) score space
        ranking = self.score_transform == "quantile"
        rescale = self.score_transform is not None
        bins = get_bins(y_pred, n_bins=self.n_bins, ranking=ranking, rescale=rescale)

        # Get Calibration Curves
        _, fraction_of_positives_dis = get_calibration_curve(y_true_dis, y_pred_dis, bins)
        _, fraction_of_positives_ad = get_calibration_curve(y_true_ad, y_pred_ad, bins)
        mean_predicted_value, fraction_of_positives = get_calibration_curve(y_true, y_pred, bins)

        # calculate differences between calibration curves (and determine positive and negative parts)
        nonzero = ~np.isnan(fraction_of_positives_ad) & ~np.isnan(fraction_of_positives_dis)
        differences = fraction_of_positives_ad[nonzero] - fraction_of_positives_dis[nonzero]

        differences_pos = differences.copy()
        differences_pos[differences_pos < 0] = 0
        differences_neg = differences.copy()
        differences_neg[differences_neg > 0] = 0

        if self.weighting == "quantiles":
            # weights = None
            # weight by number of samples per bin
            weights = [y_pred[(y_pred >= bins[0]) & (y_pred <= bins[1])].size]
            for i in range(2, len(bins)):
                weights = weights + [y_pred[(y_pred > bins[i - 1]) & (y_pred <= bins[i])].size]
            weights = np.array(weights)[nonzero]
            # (histogram does not use closed intervals like percentiles should be)
        elif self.weighting == "scores":
            # integral (weights by bin length)
            weights = np.ediff1d(bins)
            weights[0] = weights[0] - mean_predicted_value[0]
            weights = weights[nonzero]
        else:
            raise ValueError("weight_by must be 'quantiles' or 'scores'")
        positive_component = np.average(differences_pos, weights=weights)
        negative_component = np.average(differences_neg, weights=weights)

        # sum of both = integral
        net_distance = positive_component - negative_component

        return TwoGroupBiasResult(net_distance, positive_component, -1 * negative_component)


