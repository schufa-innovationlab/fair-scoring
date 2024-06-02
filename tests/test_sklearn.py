"""
Example: Tests for sklearn compatibility
"""
# TODO: remove, if there is no sklearn-estimator
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairscoring import MyFancyEstimator # TODO: add all sklearn-estimators, here

@parametrize_with_checks([
    # TODO: add all sklearn-estimators, here
    MyFancyEstimator(),
])
def test_sklearn_compatibility(estimator, check):
    check(estimator)

