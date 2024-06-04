"""
Test for table 1
"""

import pytest
import pandas as pd
import numpy as np

from fairscoring.metrics import bias_cal, bias_eo, bias_pe
from fairscoring.metrics.roc import bias_roc, bias_xroc


@pytest.fixture(scope="session")
def compas_data():
    """
    Fixture that loads the Compas Dataset
    """
    dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
    df = pd.read_csv(dataURL)

    df.rename(columns=dict((column_name, column_name.lower()) for column_name in df.columns),
              inplace=True)

    score_column = 'decile_score'
    target_column = 'two_year_recid'
    protected_attribute_column = 'race'

    # Get Columns
    scores = df[score_column]
    target = df[target_column]
    attribute = df[protected_attribute_column]

    return [scores, target, attribute]



@pytest.mark.parametrize(
    "metric,bias,pos,neg",
    [
        (bias_eo,0.161,0,100),
        (bias_pe,0.154,0,100),
        (bias_cal,0.034,79,21),
        (bias_roc,0.016,46,54),
        (bias_xroc,0.273,0,100),
    ],
    ids=["EO", "PE", "Cali", "ROC", "xROC"])
def test_compas_tab_1(compas_data, metric, bias, pos, neg):
    groups = ['African-American', 'Caucasian']
    favorable_target = 0

    scores = compas_data[0]
    target = compas_data[1]
    attr = compas_data[2]

    result = metric.bias(scores, target, attr, groups, favorable_target, prefer_high_scores=False)

    bias_ = np.round(result.bias, 3)
    pos_ = np.round(100 * result.pos_component, 0)
    neg_ = np.round(100 * result.neg_component, 0)

    assert bias_ == bias
    assert pos_ == pos
    assert neg_ == neg
