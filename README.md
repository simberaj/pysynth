# PySynth: Dataset Synthesis for Python

PySynth is a package to create synthetic datasets - that is, datasets that look
just like the original in terms of statistical properties, variable values,
distributions and correlations, but do not have exactly the same contents
so are safe against data disclosure. An alternative to R's 
[Synthpop](https://www.r-bloggers.com/generating-synthetic-data-sets-with-synthpop-in-r/)
with a more permissive license.

## Installation
You can get PySynth from PyPI by using the obvious

    pip install pysynth

## Usage
You can perform the synthesis with basic settings directly on a CSV file:

    python -m pysynth source.csv synthesized.csv

This produces a `synthesized.csv` file that will look a lot like the original
(variable names values, distributions, correlations) but will (most likely)
not be the same.

For better control, it is best to use the synthesizer objects. They follow the
scikit-learn interface for Pandas dataframes so you `fit()` them on the
original and then `sample(n)` to get a synthetic dataframe of `n` rows.

So far, only a synthesizer based on iterative proportional fitting
(`pysynth.ipf.IPFSynthesizer`) is available. This synthesis bins continuous
variables to categories and reconstructs them using fitted univariate
distributions. Missing values (`NaN`) are preserved.

Synthesis quality measurement modules to be added.

## Contributors
Feedback, additions, suggestions, issues and pull requests are welcome and much
appreciated on [GitHub](https://github.com/simberaj/pysynth).

How to add features:

1.  Fork it (https://github.com/simberaj/pysynth/fork)
2.  Create your feature branch (`git checkout -b feature/feature-name`)
3.  Commit your changes (`git commit -am "feature-name added"`)
4.  Push to the branch (`git push origin feature/feature-name`)
5.  Create a new pull request

Development requires `pytest` for testing and `sphinx` to generate
documentation. Tests can be run using simple

    pytest tests

### Intended development directions
-   Synthesis quality measurement in terms of anonymization/similarity
-   Model-based synthesis along the lines of R's Synthpop

## License and author info
PySynth is developed by Jan Šimbera <simbera.jan@gmail.com>.

PySynth is available under the MIT license. See `LICENSE.txt` for more details.
