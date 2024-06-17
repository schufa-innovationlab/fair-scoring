"""
Helpful function when dealing with fairness-related data
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, List


# Internal encoding
_ENCODING_FAVORABLE_OUTCOME = 0
_ENCODING_UNFAVORABLE_OUTCOME = 1

def split_groups(
        attribute: ArrayLike,
        groups: List
) -> List[ArrayLike]:
    """
    Split the dataset into groups.

    Parameters
    ----------
    attribute: ArrayLike
        The protected attribute. Must have the same length as `scores`.

    groups: List
        A list of groups. Each group is given by a value of the protected attribute.
        A value of `None` is used to define a group with all elements that are not in another group.

    Returns
    -------
    filter: List of ArrayLike
        Returns for each group an array binary filter, that is 'True' for those elements that belong to the group.

    """
    attribute = np.asarray(attribute)

    # Create a filter for each normal group
    filters = []
    for grp in groups:
        if grp is None:
            filters.append(np.isin(attribute, groups, invert=True))
        else:
            filters.append(attribute == grp)

    return filters


def _check_input(
        scores: ArrayLike,
        target: ArrayLike,
        attribute: ArrayLike,
        groups: list,
        favorable_target: Union[str, int]
) -> tuple[ArrayLike, ArrayLike, ArrayLike, list]:
    """
    Checks & normalizes the input values.

    Parameters
    ----------
    scores: ArrayLike
        A list of scores

    target: ArrayLike
        The binary target values. Must have the same length as `scores`.

    attribute: ndarray
        The protected attribute. Must have the same length as `scores`.

    groups: list, optional
        A list of groups. Each group is given by a value of the protected attribute.
        A value of `None` is used to define a group with all elements that are not in another group.

    favorable_target: str or int
        The favorable outcome

    Returns
    -------
    scores: ArrayLike
        A list of scores

    target: ArrayLike
        The binary target values.

    attribute: ndarray
        The protected attribute.

    groups: list
        A list of groups. Each group is given by a value of the protected attribute.
        A value of `None` is used to define a group with all elements that are not in another group.

    Raises
    ------
    TODO: define Errors
    """
    # TODO: Implement me
    # Check dimensions

    # Encode target
    encoding = {False: _ENCODING_UNFAVORABLE_OUTCOME, True: _ENCODING_FAVORABLE_OUTCOME}
    target = np.asarray(target) == favorable_target
    target = np.vectorize(encoding.get)(target)  # Apply encoding map

    # Convert to numpy
    scores = np.asarray(scores)
    attribute = np.asarray(attribute)

    return scores, target, attribute, groups
