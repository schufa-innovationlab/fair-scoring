"""
Base class for bias metrics
"""
from __future__ import annotations
import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from sklearn.preprocessing import quantile_transform
from typing import Union, Literal, Optional

from fairscoring.utils import split_groups, _check_input


class BaseBiasMetric(ABC):
    """
    Base bias metric. This class covers the basic bias computation workflow.

    Parameters
    ----------
    name: str
        Name of the metric.

    score_transform: {"rescale","quantile",None}
        A transformation of the scores prior to the bias computation.
        There are two supported methods:

        - rescaling (to the interval `[0,1]`.
          In this case, the :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias` method can take min and max scores.
        - quantile transformation. This leads to standardized bias measures.
    """
    def __init__(self,
                 name: str,
                 score_transform: Optional[Literal["rescale", "quantile"]] = None
                 ):
        self.name = name
        self.score_transform = score_transform

    def _split_groups(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: np.ndarray,
            groups: list,
            return_total: bool = False
    ) -> list[np.ndarray]:
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

        # Create a filter for each normal group
        filters = split_groups(attribute, groups)

        # Handle the case of returning all elements
        if return_total:
            filters.append(np.ones(attribute.shape, dtype=bool))

        return filters

    @abstractmethod
    def _groupwise_data (
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: ArrayLike,
            groups_filter: list[np.ndarray]
    ) -> list:
        """
        Gets groupwise data. Typically, this is an array of scores or a tuple of scores and labels.
        Which data is returned depends on the concrete bias metric

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
        data: list
            One list element per group with the relevant data.

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks need to be performed.
        """
        pass

    @abstractmethod
    def _compute_bias(self, *groups_data, min_score, max_score) -> BiasResult:
        """
        Compute the bias. This is the core metric to be implemented by specific metrics

        Parameters
        ----------
        groups_data: any
            One parameter for each group

        min_score: float
            The minimal score. This might influence the bias computation, e.g. by defining the integral bounds.

        min_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.

        Returns
        -------
        result: BiasResult
            The results of the bias computation, including intermediate values.
        """
        pass

    def _transform_scores(self, scores:ArrayLike, min_score:float=None, max_score:float=None) -> ArrayLike:
        """
        Transforms the scores.

        Parameters
        ----------
        scores: ArrayLike
            The original scores.

        Returns
        -------
        scores: ArrayLike
            The transformed scores

        min_score: float
            The minimal score after the transformation

        max_score: float
            The maximal score after the transformation

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks need to be performed.
        """
        if self.score_transform == "rescale":
            if min_score is None:
                min_score = np.min(scores)
            if max_score is None:
                max_score = np.max(scores)

            # TODO: do we need a check for out of bounds values?
            return (scores - min_score) / (max_score - min_score), 0, 1

        if self.score_transform == "quantile":
            n_quantiles = min(50000, scores.shape[0])

            # Quantile Transformer limits number of samples used for estimation. Number set to 10 Mio.
            # If the dataset is larger than that, results may slightly vary.
            # Fixed Random seed can be added here to remove this effect.
            scores = quantile_transform(X=scores.reshape(-1, 1), n_quantiles=n_quantiles, subsample=10000000)

            return scores.reshape(-1), 0, 1

    def _bias_iteration(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: ArrayLike,
            groups: list,
            min_score: float,
            max_score: float):
        """
        One iteration (in case of permutation tests)

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

        min_score: float
            The minimal score. This might influence the bias computation, e.g. by defining the integral bounds.

        min_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.

        Returns
        -------
        bias: BiasResult
            The computed bias (including intermediate results)
        """

        # Split by Groups
        # TODO: what about return_total
        grp_filters = self._split_groups(scores, target, attribute, groups)
        grp_data = self._groupwise_data(scores, target, attribute, grp_filters)

        # Compute Bias
        # TODO: is there general information that needs to be injected into the result?
        result = self._compute_bias(*grp_data, min_score=min_score, max_score=max_score)

        return result

    def bias(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: ArrayLike,
            groups: list,
            favorable_target: Union[str, int],
            *,
            min_score: Optional[float] = None,
            max_score: Optional[float] = None,
            n_permute: Optional[int] = None,
            seed:Optional[int] = None,
            prefer_high_scores: bool = True
    ) -> BiasResult:
        """
        Bias computation

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
            The favorable outcome

        min_score: float
            The minimal score. This might influence the bias computation, e.g. by defining the integral bounds.
            This is also used for rescaling.

        max_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.
            This is also used for rescaling

        n_permute: int, optional
            Number of iterations for the permutation test.
            Permutation tests are only performed if this value is `>0`.

        prefer_high_scores: bool, optional
            Specify whether high scores or low scores are favorable.

        Returns
        -------
        bias: BiasResult
            The computed bias (including intermediate results)

        Other Parameters
        ----------------
        seed: int, optional
            Random seed for the permutation test.
            Only required if the result need to be 100% reproducible.

        Notes
        -----
        This is an internal method that should not be called directly. For this reason, no checks need to be performed.
        """
        # Check & normalize Input
        scores, target, attribute, groups = self._check_input(scores, target, attribute, groups, favorable_target)

        # Filter out non-group-elements
        if None not in groups:
            # Only filter, if no "all-others"-group (indicated by None) is set.
            filter = np.isin(attribute, groups)
            scores = scores[filter]
            target = target[filter]
            attribute = attribute[filter]

        # Preprocess scores
        if self.score_transform is not None:
            scores, min_score, max_score = self._transform_scores(scores, min_score=min_score, max_score=max_score)
        else:
            # Otherwise compute bounds
            if min_score is None:
                min_score = np.min(scores)
            if max_score is None:
                max_score = np.max(scores)

        # Internal score orientation to high_score_is_good:
        if not prefer_high_scores:
            scores = max_score + min_score - scores

        result = self._bias_iteration(scores, target, attribute, groups, min_score, max_score)

        if n_permute is not None and n_permute > 0:
            rng = np.random.default_rng(seed)
            # Compute the bias n_perm times with permuted attribute array.
            bias_perm = np.asarray([self._bias_iteration(scores, target, rng.permutation(attribute), groups, min_score, max_score).bias
                                    for _ in range(n_permute)])
            p_value = (1 + np.sum(bias_perm > result.bias)) / (1 + n_permute)

            result.p_value = p_value

        # Return result
        return result

    def __call__(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: ArrayLike,
            groups: list,
            favorable_target: Union[str, int],
            *,
            min_score: Optional[float] = None,
            max_score: Optional[float] = None,
            prefer_high_scores: bool = True
    ) -> float:
        """
        Bias computation.

        This method allows to use the bias metric as a function.

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
            The favorable outcome

        min_score: float
            The minimal score. This might influence the bias computation, e.g. by defining the integral bounds.
            This is also used for rescaling.

        max_score: float
            The maximal score. This might influence the bias computation, e.g. by defining the integral bounds.
            This is also used for rescaling

        prefer_high_scores: bool, optional
            Specify whether high scores or low scores are favorable.

        Returns
        -------
        bias: float
            The computed bias.

        Notes
        -----
        This method offers fewer parameters than :meth:`~fairscoring.metrics._base.BaseBiasMetric.bias`,
        because not all will affect the pure bias value.
        """
        # Compute bias
        result = self.bias(scores, target, attribute, groups, favorable_target, min_score=min_score, max_score=max_score,
                           prefer_high_scores=prefer_high_scores)

        # Only return the pure bias value
        return result.bias

    def _check_input(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            attribute: ArrayLike,
            groups: list,
            favorable_target: Union[str, int]
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, list]:  # TODO: Specify
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

        groups: list
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
        # The base function just calls an external helper function
        return _check_input(scores,target,attribute,groups,favorable_target)


class BiasResult:
    """
    A base class to store bias results.

    Child classes of this class are used to store metric-specific intermediate results that allow for further
    bias analyses, such as metric-specific plots.

    Parameters
    ----------
    bias: float
        The bias value

    Attributes
    ----------
    bias: float
        The bias value

    p_value: float, optional
        The p_value of statistical tests.
    """
    def __init__(self, bias:float):
        self.bias = bias
        self.p_value = None


class TwoGroupMixin(BaseBiasMetric):
    """
    A mixin for metrics that only support two groups
    """

    def _check_input(self, scores: ArrayLike, target: ArrayLike, attribute: ArrayLike, groups: list,
                     favorable_target: Union[str, int]) -> tuple[ArrayLike, ArrayLike, ArrayLike, list]:
        # Check number of groups
        n = len(groups)
        if n != 2:
            raise ValueError(f"Metric requires exactly two groups. Instead {n} groups where provided.")

        return super()._check_input(scores, target, attribute, groups, favorable_target)

    @abstractmethod
    def _compute_bias(self, disadv_grp, adv_grp, total=None, min_score=None, max_score=None) -> TwoGroupBiasResult:
        pass


class TwoGroupBiasResult(BiasResult):
    """
    A base class to store bias results.

    Child classes of this class are used to store metric-specific intermediate results that allow for further
    bias analyses, such as metric-specific plots.

    Parameters
    ----------
    bias: float
        The bias value

    pos: float
        The positive component of the bias

    neg: float
        The negative component of the bias

    Attributes
    ----------
    bias: float
        The bias value

    pos: float
        The positive component of the bias

    neg: float
        The negative component of the bias
    """
    def __init__(self, bias: float, pos: float, neg: float):
        super().__init__(bias=bias)
        self.pos = pos
        self.neg = neg

    @property
    def pos_component(self):
        """
        Proportion of the positive component in the total bias

        Returns
        -------
        Proportion of the positive component in the total bias
        """
        return np.abs(self.pos / self.bias)

    @property
    def neg_component(self):
        """
        Proportion of the negative component in the total bias

        Returns
        -------
        Proportion of the negative component in the total bias
        """
        return np.abs(self.neg / self.bias)


