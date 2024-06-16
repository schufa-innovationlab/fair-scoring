import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.calibration import calibration_curve

from fairscoring.plots._utils import lighten_color
from fairscoring.utils import split_groups

from numpy.typing import ArrayLike
from typing import Union, Optional, List


def plot_groupwise_score_calibration(
        scores: ArrayLike,
        target: ArrayLike,
        attribute: ArrayLike,
        groups: Optional[List] = None,
        palette: Union[dict,list] = sns.color_palette(),
        n_bins: int = 20,
        n_bootstrap: int = 10,
        strategy: str = 'uniform',
        rescale: bool = True,
        rescale_by: Optional[List] = None,
        figsize: tuple = (10, 5)):
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

    palette :
        color palette, number of colors must equal the categories of y

    n_bins : int
        number of bins

    n_bootstrap : int, optional
        number of bootstrap samples; can be slow for large datasets. If None, no bootstrapping is performed.

    strategy : {'uniform', 'quantiles'}
        strategy to determine bins

    rescale_by
        maximum and minimum possible score value for rescaling. If None, the minimum and maximum from the data are used.

    rescale
        True if the values are rescaled

    figsize: tuple
        Size of the figure.

    See Also
    --------
    fairscoring.metrics.CalibrationMetric
    sklearn.calibration.calibration_curve
    """
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

    # Create figure &
    plt.figure(figsize=figsize)
    plt.plot([1, 0], [0, 1], c='black')

    # Iterate Groups
    for i, grp_flt in enumerate(split_groups(attribute, groups)):
        idx, = np.nonzero(grp_flt)
        grp = groups[i]

        if isinstance(palette, dict):
            color = palette[grp]
        else:
            color = palette[i]

        if n_bootstrap is not None:
            # Bootstrap Colors are brighter
            color_light = lighten_color(color, amount=0.5)

            for i in range(n_bootstrap):
                # Bootstrap Sampling
                idx_boot = np.random.choice(idx, size=len(idx), replace=True)

                # Compute & Plot Bootstrap Calibration Curve
                prob_true, prob_pred = calibration_curve(target[idx_boot], (scores[idx_boot] - rescale_by[0]) / (rescale_by[1] - rescale_by[0]),
                                                         n_bins=n_bins, strategy=strategy)
                plt.plot(prob_pred, prob_true, c=color_light, linewidth=1)

        # Compute & Plot Bootstrap Calibration Curve
        prob_true, prob_pred = calibration_curve(target[idx], (scores[idx] - rescale_by[0]) / (rescale_by[1] - rescale_by[0]),
                                                 n_bins=n_bins, strategy=strategy)
        plt.plot(prob_pred, prob_true, c=color, marker='o', linewidth=1)

    # Label Plots
    plt.xlabel('score')
    plt.ylabel('true rate')
    plt.title('Score Calibration')

    plt.show()
