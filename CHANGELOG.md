# Change Log

## [0.2.0] - 20??-??-??

### Added
#### Plots
- Calibration plot `fairscoring.plots.plot_groupwise_score_calibration` to visualize calibration bias.
- Cumulative distribution plot `fairscoring.plots.plot_groupwise_cdfs` and the difference `fairscoring.plots.plot plot_cdf_diffs`
  to visualize independence, equal opportunity and predictive equality bias.
- Custom colormaps in `fairscoring.plots.colors`

#### Bias Result Objects
- New result type `IntegralBiasResult` now contain the cdfs of the group distributions.

### Changed
- Submodules of `fairscoring.metrics` are now public.
- The class `TwoGroupMixin` is now called `TwoGroupMetric`. 
  This allows for a more consistent naming scheme.

## [0.1.1] - 2024-06-19
 
### Added
- Parameter `prefer_high_scores` to allow for both types of scores.
  This allowed for shorter dataset handling of the COMPAS examples.
- Preparation to host docs on readthedocs. 
 
### Fixed
 
- Show correct project-link on [pypi.org](https://pypi.org/project/fair-scoring/) 
- `README.md` shows the correct installation command

## [0.1.0] - 2024-06-03
 
### Added
- Bias Metric Classes for
  - Calibration: `fairscoring.metrics.CalibrationMetric`
  - Wasserstein Distance: `fairscoring.metrics.WassersteinMetric`
  - ROC-base Metrics: `fairscoring.metrics.roc.ROCBiasMetric`
- Default Bias Metrics for
  - Equal Opportunity: `fairscoring.metrics.bias_eo`
  - Predictive Equality: `fairscoring.metrics.bias_pe`
  - Calibration: `fairscoring.metrics.bias_cal`
  - Independence: `fairscoring.metrics.bias_ind`
  - ROC: `fairscoring.metrics.roc.bias_roc`
  - xROC: `fairscoring.metrics.roc.bias_xroc`
- Experiments on
  - Adult Dataset
  - COMPAS Dataset
  - German Credit Risk Dataset
- Tests for compatibility with results in the publication
  - COMPAS Dataset