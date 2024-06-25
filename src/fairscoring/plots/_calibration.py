import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

import numpy as np
from sklearn.calibration import calibration_curve

from fairscoring.utils import split_groups, _check_input

from numpy.typing import ArrayLike
from typing import Union, Optional, List


def plot_groupwise_score_calibration(
        scores: ArrayLike,
        target: ArrayLike,
        attribute: ArrayLike,
        groups: List,
        favorable_target: Union[str, int],
        *,
        ax: Optional[matplotlib.axes.Axes] = None,
        palette: Union[dict,list] = sns.color_palette(),
        n_bins: int = 20,
        n_bootstrap: int = 10,
        rescale: bool = True,
        rescale_by: Optional[List] = None,
        prefer_high_scores: bool = True,
        strategy: str = 'uniform'):
    """

    Plot groupwise score calibration.
    Calibration plots. Optional with bootstrap curves.

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

    favorable_target: str or int
        The favorable outcome.

    palette : dict or list, Optional
        Color palette, number of colors must equal the categories of y

    n_bins : int, Default=20
        Number of bins

    n_bootstrap : int, optional, Default=10
        Number of bootstrap samples; can be slow for large datasets.
        Set to `None`, to disable bootstrapping.

    rescale_by: List, optional
        Maximum and minimum possible score value for rescaling. If None, the minimum and maximum from the data are used.

    rescale: bool, default=True
        True if the values are rescaled.

    prefer_high_scores: bool, optional
        Specify whether high scores or low scores are favorable.

    Other Parameters
    ----------------
    strategy : {'uniform', 'quantiles'}
        Strategy to determine bins


    See Also
    --------
    fairscoring.metrics.CalibrationMetric
    sklearn.calibration.calibration_curve
    """
    scores, target, attribute, groups = _check_input(scores, target, attribute, groups, favorable_target)

    # Scaling boundaries.
    if rescale:
        if rescale_by is None:
            rescale_by = [min(scores), max(scores)]
        elif len(rescale_by) != 2:
            raise ValueError("rescale_by must be of length 2 and contain minimum and maximum score.")
        elif rescale_by[0] == rescale_by[1]:
            raise ValueError("Minimum and maximum score cannot be identical.")
    else:
        rescale_by = [0, 1]

    # If smaller scores are better, then swap order when rescaling
    if not prefer_high_scores:
        rescale_by = [rescale_by[1], rescale_by[0]]

    # Rescale scores
    scores = (scores - rescale_by[0]) / (rescale_by[1] - rescale_by[0])

    # Helper Function project scaled values back to the original Space
    def inverse_scaling(scores):
        return np.asarray(scores) * (rescale_by[1] - rescale_by[0]) + rescale_by[0]

    # Handle defaults
    if ax is None:
        ax = plt.gca()

    x_limits = inverse_scaling([1, 0])
    ax.plot(x_limits, [0, 1], c='black')

    # Iterate Groups
    for i, grp_flt in enumerate(split_groups(attribute, groups)):
        # Filter Groups
        idx, = np.nonzero(grp_flt)
        grp = groups[i]

        # Pick color by name or index
        if isinstance(palette, dict):
            color = palette[grp]
        else:
            color = palette[i]

        if n_bootstrap is not None:
            for i in range(n_bootstrap):
                # Bootstrap Sampling
                idx_boot = np.random.choice(idx, size=len(idx), replace=True)

                # Compute Bootstrap Calibration Curve
                prob_true, prob_pred = calibration_curve(target[idx_boot], scores[idx_boot], n_bins=n_bins, strategy=strategy)

                # Undo scaling for original scores on x-axis
                prob_pred = inverse_scaling(prob_pred)

                ax.plot(prob_pred, prob_true, c=color, alpha=0.5, linewidth=1)

        # Compute Calibration Curve
        prob_true, prob_pred = calibration_curve(target[idx], scores[idx], n_bins=n_bins, strategy=strategy)

        # Undo scaling for original scores on x-axis
        prob_pred = inverse_scaling(prob_pred)
        ax.plot(prob_pred, prob_true, c=color, marker='o', linewidth=1, label=grp)

    # Label Plots
    ax.set_xlabel('score')
    ax.set_ylabel('true rate')
    ax.set_title('Score Calibration')
