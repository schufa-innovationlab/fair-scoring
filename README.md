# Fair-Scoring
Fairness metrics for continuous risk scores.

The implemented algorithms are described in the paper [[1]](#References). 

## Installation

Install with `pip` directly:
```shell
pip install fair-scoring
```

## Usage
The following example shows how compute the equal opportunity bias of the compas dataset

```python
import pandas as pd
from fairscoring.metrics import bias_eo

# Load compas data
dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(dataURL)

# Relevant data
scores = df['decile_score']
target = df['two_year_recid']
attribute = df['race']

# Compute the bias
bias = bias_eo(scores, target, attribute, groups=['African-American', 'Caucasian'],favorable_target=0,prefer_high_scores=False)
```

### Further examples
Further examples - especially the experiments conducted for the publication -  can be found 
[in the documentation](docs/source/examples).

## Development
### Setup
Clone the repository and install from this source via

```shell
pip install -e .[dev]
```

### Tests
To execute the tests install the package in development mode (see above)
```
pytest
```

Following the pytest framework, tests for each package are located in a subpackages named `test`

### Docs
To build the docs move to the `./docs` subfolder and call
```shell
make clean
make html
```

## References
[1] Becker, A.K. and Dumitrasc, O. and Broelemann, K.;
Standardized Interpretable Fairness Measures for Continuous Risk Scores;
Proceedings of the 41th International Conference on Machine Learning, 2024;
<details><summary>Bibtex</summary>
<p>

```
@inproceedings{Zern2023Interventional,
    author = {Ann{-}Kristin Becker and Oana Dumitrasc and Klaus Broelemann}
    title  = {Standardized Interpretable Fairness Measures for Continuous Risk Scores},
    booktitle={Proceedings of the 41th International Conference on Machine Learning},
    year = {2024}
}
```

</p>
</details>
