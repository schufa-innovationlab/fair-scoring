"""
Helpful function when dealing with fairness-related data
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, List


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
