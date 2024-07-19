import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


import numpy as np
from fairscoring.metrics import WassersteinMetric

from numpy.typing import ArrayLike
from typing import Union, List, Tuple, Literal, Optional


def plot_groupwise_cdfs(
        scores: ArrayLike,
        target: ArrayLike,
        attribute: ArrayLike,
        groups: List,
        favorable_target: Union[str, int],
        *,
        ax: Optional[matplotlib.axes.Axes] = None,
        palette: Union[list,matplotlib.colors.Colormap] = sns.color_palette(),
        fairness_type:Literal["IND", "EO", "PE"],
        quantile_transform: bool = True,
        prefer_high_scores: bool = True):
    """
    Plot groupwise cumulative distributions functions (cdfs).

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

        __Note__: Currently, only plots with exactly two groups are possible.

    favorable_target: str or int
        The favorable outcome.

    ax: matplotlib.axes.Axes, optional
        The axes into which the cdfs shall be plotted

    palette : list or Colormap, Optional
        Color palette, number of colors must at least be number of groups

    fairness_type: {"IND", "EO", "PE"}
        Specifies the type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality),

    quantile_transform: bool, default=True
        Specify whether the scores shall be quantile transformed.

    prefer_high_scores: bool, default=True
        Specify whether high scores or low scores are favorable.

    """
    if quantile_transform:
        score_transform = "quantile"
    else:
        score_transform = None

    # Apply Metric
    # TODO: better name handling. Also include this as title into the plots
    metric = WassersteinMetric(fairness_type=fairness_type, score_transform=score_transform, name="Wasserstein Bias")
    result = metric.bias(scores, target, attribute, groups, favorable_target, prefer_high_scores=prefer_high_scores)

    _plot_cdfs(result.cdf_x, result.cdfs, groups=groups,
               ax=ax, palette=palette,
               fairness_type=fairness_type, score_transform=score_transform, prefer_high_scores=prefer_high_scores)


def plot_cdf_diffs(
        scores: ArrayLike,
        target: ArrayLike,
        attribute: ArrayLike,
        groups: List,
        favorable_target: Union[str, int],
        *,
        ax: Optional[matplotlib.axes.Axes] = None,
        palette: Union[list,matplotlib.colors.Colormap] = sns.color_palette(),
        fairness_type:Literal["IND", "EO", "PE"],
        quantile_transform: bool = True,
        prefer_high_scores: bool = True):
    """
    Plots the difference between two cumulative distributions functions (cdfs).

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

        __Note__: There must be exactly two groups to plot the differences.

    favorable_target: str or int
        The favorable outcome.

    ax: matplotlib.axes.Axes, optional
        The axes into which the cdfs shall be plotted

    palette : list or Colormap, Optional
        Color palette, number of colors must at least be number of groups

    fairness_type: {"IND", "EO", "PE"}
        Specifies the type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality),

    quantile_transform: bool, default=True
        Specify whether the scores shall be quantile transformed.

    prefer_high_scores: bool, default=True
        Specify whether high scores or low scores are favorable.

    """
    if quantile_transform:
        score_transform = "quantile"
    else:
        score_transform = None

    # Apply Metric
    # TODO: better name handling. Also include this as title into the plots
    metric = WassersteinMetric(fairness_type=fairness_type, score_transform=score_transform, name="Wasserstein Bias")
    result = metric.bias(scores, target, attribute, groups, favorable_target, prefer_high_scores=prefer_high_scores)

    _plot_cdf_diffs(result.cdf_x, result.cdfs, groups=groups,
               ax=ax, palette=palette,
               fairness_type=fairness_type, score_transform=score_transform, prefer_high_scores=prefer_high_scores)


def _plot_cdfs(
        cdf_x: ArrayLike,
        cdfs: ArrayLike,
        groups: Optional[list] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        palette: Union[list,matplotlib.colors.Colormap] = sns.color_palette(),
        prefer_high_scores: bool = True,
        fairness_type: Optional[Literal["IND", "EO", "PE"]] = None,
        score_transform: Optional[Literal["rescale", "quantile"]] = None,
        scaled_from: Optional[Tuple[float, float]] = None,
):
    """
    Inner function for plotting cumulative distributions functions (cdfs).

    Parameters
    ----------
    cdf_x: ArrayLike
        x-values at which the cdfs are stored. This array is 1-dimensional

    cdfs: List of ArrayLike
        List of cdfs, where each cdf ist stored as array of cdf-values corresponding to the values of `cdf_x`.

        __Note__: We assume that the cdfs are based on preprocessed scores, so that higher scores are preferred.
        We use `prefer_high_scores` to undo that effect in the plots.

    groups: list, optional
        A list of groups. Each group is given by a value of the protected attribute.
        The groups are used to label groups in the plot.

    ax: matplotlib.axes.Axes, optional
        The axes into which the cdfs shall be plotted

    palette : list or Colormap, Optional
        Color palette, number of colors must at least be number of groups

    Other Parameters
    ----------------
    prefer_high_scores: bool, default=True
        Specify whether high scores or low scores are favorable.
        If set to true, the plot will show ``1-cdf``, i.e. the acceptance rate.

        __Note__: There is a connection / interaction with the parameter `scaled_from`:

        - If `scaled_from` is set to ``(worst_score,best_score)`` with ``worst_score > best_score`` then
          `prefer_high_scores` is automatically ``False``
        - If ``prefer_high_scores=False`` and `scaled_from` is not set, all score values (i.e. `cdf_x`) are reverted

    fairness_type: {"IND", "EO", "PE"}, optional
        The type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality)

        This parameter is used to determine the label of the y-axis

    score_transform: {"rescale","quantile",None}
        The transformation applied to the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`).
          In this case, the :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.

        This parameter is used to determine the label of the x-axis

    scaled_from: (float, float), optional
        Pair with ``(min_score, max_score)`` values.

        It is possible to revert the score order. This can either be done by having ``min_score > max_score`` or
        by not providing `scaled_from` and setting `prefer_high_scores`.

        See also the description of `prefer_high_scores` for further details.

    """
    # TODO: Check input

    # Handle prefer_high_scores / scaled_from dependencies
    cdf_x, x_label = _preprocess_x_values(cdf_x, prefer_high_scores, score_transform, scaled_from)

    # Handle defaults
    if ax is None:
        ax = plt.gca()

    if groups is None:
        groups = [None] * cdfs.shape[1]

    for i, grp in enumerate(groups):
        # Pick color
        if isinstance(palette, matplotlib.colors.Colormap):
            color = palette(i)
        else:
            color = palette[i]

        # Acceptance rate
        # Note: if prefer_high_scores==False, then reverting of the x-axis already does the trick.
        # The y-axis must not be treated differently.
        ax.step(cdf_x, 1 - cdfs[i], c=color, label=grp)

    # Plot the legend
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(_get_y_label(fairness_type, is_diff=False))


def _plot_cdf_diffs(
        cdf_x: ArrayLike,
        cdfs: List[ArrayLike],
        groups: Optional[list] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        palette: Union[list,matplotlib.colors.Colormap] = sns.color_palette(),
        prefer_high_scores: bool = True,
        fairness_type: Optional[Literal["IND", "EO", "PE"]] = None,
        score_transform: Optional[Literal["rescale", "quantile"]] = None,
        scaled_from: Optional[Tuple[float, float]] = None,
):
    """
    Inner function for plotting cumulative distributions functions (cdfs).

    Parameters
    ----------
    cdf_x: ArrayLike
        x-values at which the cdfs are stored. This array is 1-dimensional

    cdfs: List of ArrayLike
        List of exactly two cdfs, where each cdf ist stored as array of cdf-values corresponding to the values of `cdf_x`.

        __Note__: We assume that the cdfs are based on preprocessed scores, so that higher scores are preferred.
        We use `prefer_high_scores` to undo that effect in the plots.

    groups: list
        A list of groups. Each group is given by a value of the protected attribute.

        __Note__: There must be exactly two groups to plot the differences.

    ax: matplotlib.axes.Axes, optional
        The axes into which the cdfs shall be plotted

    palette : list or Colormap, Optional
        Color palette, number of colors must at least be number of groups

    Other Parameters
    ----------------
    prefer_high_scores: bool, default=True
        Specify whether high scores or low scores are favorable.
        If set to true, the plot will show ``1-cdf``, i.e. the acceptance rate.

        __Note__: There is a connection / interaction with the parameter `scaled_from`:

        - If `scaled_from` is set to ``(worst_score,best_score)`` with ``worst_score > best_score`` then
          `prefer_high_scores` is automatically ``False``
        - If ``prefer_high_scores=False`` and `scaled_from` is not set, all score values (i.e. `cdf_x`) are reverted

    fairness_type: {"IND", "EO", "PE"}, optional
        The type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality)

        This parameter is used to determine the label of the y-axis

    score_transform: {"rescale","quantile",None}
        The transformation applied to the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`).
          In this case, the :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.

        This parameter is used to determine the label of the x-axis

    scaled_from: (float, float), optional
        Pair with ``(min_score, max_score)`` values.

        It is possible to revert the score order. This can either be done by having ``min_score > max_score`` or
        by not providing `scaled_from` and setting `prefer_high_scores`.

        See also the description of `prefer_high_scores` for further details.

    Raises
    ------
    ValueError
        If there are not exactly 2 cdfs.

    """
    # TODO: Check input

    if len(cdfs) != 2:
        raise ValueError(f"Expecting 2 cdfs, but found {len(cdfs)}. Plotting differences is not possible.")

    # Handle prefer_high_scores / scaled_from dependencies
    cdf_x, x_label = _preprocess_x_values(cdf_x, prefer_high_scores, score_transform, scaled_from)

    # Handle defaults
    if ax is None:
        ax = plt.gca()

    if groups is None:
        groups = ["1", "2"]

    # Plot diff-curve
    diff = (1 - cdfs[0]) - (1 - cdfs[1])

    ax.step(cdf_x, diff, c="black")

    # Pick color
    if not isinstance(palette, matplotlib.colors.Colormap):
        palette = matplotlib.colors.ListedColormap(palette)

    # Fill Pro Group 0
    ax.fill_between(
            cdf_x, 0, diff,
            where=diff > 0, alpha=0.5, color=palette(0), step='pre',
            label=f'Pro Group {groups[0]}')

    # Fill Pro Group 0
    ax.fill_between(
        cdf_x, 0, diff,
        where=diff <= 0, alpha=0.5, color=palette(1), step='pre',
        label=f'Pro Group {groups[1]}')

    # Plot the legend
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(_get_y_label(fairness_type, is_diff=True))


def _preprocess_x_values(cdf_x, prefer_high_scores, score_transform, scaled_from):
    """
    Preprocess the x-values.

    Preprocessing includes undo scaling and reverting order if lower scores are preferred

    Parameters
    ----------
    cdf_x: ArrayLike
        x-values at which the cdfs are stored. This array is 1-dimensional

    prefer_high_scores: bool, default=True
        Specify whether high scores or low scores are favorable.
        If set to true, the plot will show ``1-cdf``, i.e. the acceptance rate.

        __Note__: There is a connection / interaction with the parameter `scaled_from`:

        - If `scaled_from` is set to ``(worst_score,best_score)`` with ``worst_score > best_score`` then
          `prefer_high_scores` is automatically ``False``
        - If ``prefer_high_scores=False`` and `scaled_from` is not set, all score values (i.e. `cdf_x`) are reverted

    score_transform: {"rescale","quantile",None}
        The transformation applied to the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`).
          In this case, the :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.

        This parameter is used to determine the label of the x-axis

    scaled_from: (float, float), optional
        Pair with ``(min_score, max_score)`` values.

        It is possible to revert the score order. This can either be done by having ``min_score > max_score`` or
        by not providing `scaled_from` and setting `prefer_high_scores`.

        See also the description of `prefer_high_scores` for further details.

    Returns
    -------
    cdf_x: ArrayLike
        The modified x-values

    x_label: str
        Label for the x-axis
    """
    if not prefer_high_scores:
        if scaled_from is None:
            if score_transform is not None:
                # Transformation normalize to [0,1], so we revert the scores within this interval
                cdf_x = 1 - cdf_x
            else:
                # Revert within the observed range so that (a,b) is mapped to (b,a)
                cdf_x = np.min(cdf_x) + np.max(cdf_x) - cdf_x

    # Undo scaling for original range
    if scaled_from is not None:
        cdf_x = cdf_x * (scaled_from[1] - scaled_from[0]) + scaled_from[0]

    # Label Plot an Axis
    known_limits = (scaled_from is not None)
    x_label = _get_x_label(score_transform, known_limits=known_limits)

    return cdf_x, x_label


def _get_x_label(score_transform: Optional[Literal["rescale", "quantile"]] = None, known_limits=False):
    """
    Gets the default y-label for a fairness type

    Parameters
    ----------
    score_transform: {"rescale","quantile",None}
        The transformation applied to the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`).
          In this case, the :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.

    known_limits: bool, default = False
        If the limits are known then rescaling was undone.
        This influences the label.

    Returns
    -------
    x_label: str
        The suggested label for the y--axis
    """
    if known_limits:
        return "Score"

    if score_transform == "quantile":
        x_label = 'Score Quantiles'
    elif score_transform == "rescale" and not known_limits:
        x_label = 'Rescaled Score'
    else:
        x_label = "Score"

    return x_label

def _get_y_label(fairness_type: Optional[Literal["IND", "EO", "PE"]] = None, is_diff:bool=False):
    """
    Gets the y-label for a fairness type

    Parameters
    ----------
    fairness_type: {"IND", "EO", "PE"}, optional
        The type of fairness that is measured. Accepted values are:
        1. `"IND"` (Independence),
        2. `"EO"` (Equal Opportunity),
        3. `"PE"` (Predictive Equality)

    is_diff: bool
        Specify whether the plot shows differences or original curves

    Returns
    -------
    y_label: str
        The suggested label for the y--axis
    """

    _RATE_NAMES = {
        "IND": "Positive Rate",
        "EO": "True Positive Rate",
        "PE": "False Positive Rate"
    }
    y_label = _RATE_NAMES.get(fairness_type, "Probability")

    if is_diff:
        y_label = y_label + " Difference"
    return y_label


