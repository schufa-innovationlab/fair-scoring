# Fair-Scoring
Fairness metrics for continuous scores

## Installation

Install with `pip` directly:
```shell
pip install git+https://github.com/schufa-innovationlab/fairscoring.git
```

To install a specific version, use
```shell
pip install git+https://github.com/schufa-innovationlab/fairscoring.git@0.0.1
```
where `0.0.1` has to be replaced with any tag or branch.



## Usage
__TODO__: include a simple usage example if possible
```python
from fairscoring import some_module
# ...
```

## Development


### Setup
Clone the repository and install from this source via

```shell
pip install .[dev]
```

### Tests
To execute the tests install the package in development mode (see above)
```
pytest
```

To run performance tests, allow them explicitly and capture their print output via:
```
pytest --runslow -s
```

Following the pytest framework, tests for each package are located in a subpackages named `test`

### Docs
To build the docs move to the `./docs` subfolder and call
```shell
make clean
make html
```
