'''Bin continuous variables to categorical and reconstruct them back.

An auxiliary module that enables categorical-only synthesizers to work with
continuous variables by binning them to categories for the synthesis while
remembering the value distributions within each category, and then converting
the synthesized categories back to continuous values using those distributions.

The main work is done by the :class:`Categorizer` that does this trick for a
single variable (pandas Series). It might be further configured by using an
appropriate *binner* such as :class:`QuantileBinner` to choose the numeric
bounds for the categories
and an appropriate *distributor* such as :class:`FittingDistributor`
to remember and regenerate the intra-category value distribution.
'''

from __future__ import annotations
# from typing import Union, Optional, List, Dict, Tuple, Protocol
from typing import Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats

np.seterr(all='raise')


# class Binner(Protocol):
class Binner:
    '''Interface for numeric variable interval boundary determiners.'''
    def get(self, data: pd.Series) -> List[float]:
        '''Return a list of right-inclusive cut values, without endpoints.'''
        raise NotImplementedError


class QuantileBinner(Binner):
    '''A binner that gives quantile cuts.
    
    :param bins: Number of quantiles to bin to.
    '''
    def __init__(self, bins: int):
        self.bins = bins

    def get(self, data: pd.Series) -> List[float]:
        return data.quantile(
            (np.arange(self.bins - 1) + 1) / self.bins
        ).tolist()


class EqualRangeBinner(Binner):
    '''A binner that gives equal-range cuts.
    
    :param bins: Number of bins to bin to.
    '''
    def __init__(self, bins: int):
        self.bins = bins

    def get(self, data: pd.Series) -> List[float]:
        return np.linspace(data.min(), data.max(), self.bins + 1)[1:-1].tolist()


class AprioriBinner(Binner):
    '''A dummy binner that returns cut values it was initialized with.'''
    def __init__(self, bins: List[float]):
        self.bins = bins

    def get(self, data: pd.Series) -> List[float]:
        return self.bins


BINNERS = {
    'quantile': QuantileBinner,
    'equalrange': EqualRangeBinner,
}


# class Distributor(Protocol):
class Distributor:
    '''Interface for numeric variable reconstructors.

    Fits itself on values for a single interval, and reproduces the
    distribution for a given number of output values by random sampling.
    '''
    def copy(self) -> Distributor:
        raise NotImplementedError

    def fit(self, values: np.ndarray) -> None:
        '''Fit a distribution on the values for a given interval.'''
        raise NotImplementedError

    def generate(self, n: int) -> np.ndarray:
        '''Generate a given count of random values from the fitted distribution.'''
        raise NotImplementedError


class MeanDistributor:
    '''Reproduce the values as a constant value of their mean.'''
    def __init__(self, seed = None):
        pass

    def copy(self) -> MeanDistributor:
        return MeanDistributor()

    def fit(self, values: np.ndarray) -> None:
        self.mean = values.mean()

    def generate(self, n: int) -> np.ndarray:
        return np.full(n, self.mean)


class FittingDistributor:
    '''Reproduce the values from a univariate distribution fitted to the originals.
    
    If the input values are integers with unique value count less than
    `min_unique_continuous`, reproduce the values using random sampling.
    Otherwise, find the continuous distribution from a provided list that
    approximates the distribution of the input values the best according to
    the Kolmogorov-Smirnov two-sample statistic, fit its parameters and sample
    from it.
    
    :param cont_dists: `scipy.stats`-like continuous distributions. Need to
        support a class method `fit()` that produces all required constructor
        arguments as a tuple, and a `rvs(int)` method to generate random
        samples. Defaults to `DEFAULT_CONTINUOUS`.
    :param min_unique_continuous: Minimum number of unique values in the input
        to regard an integer distribution as continuous and not discrete.
    :param min_samples: Minimum number of generated samples to use when
        evaluating the KS fit statistic.
    :param seed: Random generator seed, applied both before fitting and before
        each generator run.
    '''
    DEFAULT_CONTINUOUS: List[scipy.stats._distn_infrastructure.rv_continuous] = [
        scipy.stats.uniform,
        scipy.stats.truncnorm,
        scipy.stats.truncexpon,
    ]

    def __init__(self,
                 cont_dists: List[scipy.stats._distn_infrastructure.rv_continuous] = DEFAULT_CONTINUOUS,
                 min_unique_continuous: int = 10,
                 min_samples: int = 100,
                 seed: Optional[int] = None,
                 ):
        self.cont_dists = cont_dists
        self.min_unique_continuous = min_unique_continuous
        self.min_samples = min_samples
        self.seed = seed

    def copy(self) -> FittingDistributor:
        return FittingDistributor(
            self.cont_dists,
            self.min_unique_continuous,
            self.min_samples,
            self.seed,
        )

    def fit(self, values: np.ndarray) -> None:
        np.random.seed(self.seed)
        is_discrete = (
            np.issubdtype(values.dtype, np.integer)
            and len(np.unique(values)) < self.min_unique_continuous
        )
        if is_discrete:
            self.generator = self._fit_discrete(values)
        else:
            self.generator = self._fit_continuous(values)
    
    def _fit_discrete(self, values: np.ndarray) -> Callable[int]:
        valcounts = pd.Series(values).value_counts()
        targets = valcounts.index.values
        probs = valcounts.values.astype(float)
        probs /= probs.sum()
        return lambda n: np.random.choice(targets, size=n, p=probs)
    
    def _fit_continuous(self, values: np.ndarray) -> Callable[int]:
        minval = values.min()
        valrange = values.max() - minval
        if valrange == 0:
            # does not matter what goes here, will be multiplied by zero anyway
            best_distro = scipy.stats.norm(0, 1)
        else:
            best_distro = None
            normalized = (values - minval).astype(float) / valrange
            best_fit = 1
            test_size = max(len(normalized), self.min_samples)
            for distro in self.cont_dists:
                args = distro.fit(normalized)
                if np.isnan(args).any(): # invalid distribution estimated
                    continue
                distro_obj = distro(*args)
                fit_est = scipy.stats.ks_2samp(
                    normalized,
                    distro_obj.rvs(test_size)
                )[0]
                if fit_est < best_fit:
                    best_distro = distro_obj
                    best_fit = fit_est
            if best_distro is None:
                raise ValueError('no distribution could be estimated')
        return lambda n: minval + valrange * best_distro.rvs(n)

    def generate(self, n: int) -> np.ndarray:
        np.random.seed(self.seed)
        return self.generator(n)



DISTRIBUTORS = {
    'mean': MeanDistributor,
    'fit': FittingDistributor,
}


class Categorizer:
    '''Convert continuous variables to categorical.

    Able to reconstruct variables to their continuous form by estimating
    distributions within bins. Categorical variables are passed through
    unchanged.

    :param binner: Method to use to determine interval boundaries for
        conversion of numeric variables to categorical. Use a :class:`Binner`
        instance or one of the following strings:

        -   `'quantile'` for binning into quantiles (:class:`QuantileBinner`),
        -   `'equalrange'` for binning into equally sized bins
            (:class:`EqualRangeBinner`).

    :param bins: Number of intervals to which to bin non-categorical variables,
        or boundaries of the intervals as a list. If a list is given, it
        overrides the *binner* argument and uses :class:`AprioriBinner`. In
        that case, do not specify the minimum or maximum in the list, just the
        intermediate cuts.
    :param max_num_cats: Maximum number of categories to accept. High numbers
        of categories make the synthesizer unstable. If the
        variable has more distinct values than this number after
        categorization, a ValueError is raised.
    :param distributor: Method to use to reconstruct numeric values for a given
        category. Use a Distributor instance
        or one of the following strings:

        - `'mean'` for :class:`MeanDistributor` (constant mean value),
        - `'fit'` for :class:`FittingDistributor` (estimated distribution).

    :param seed: Seed for the variable reconstruction.
    '''
    def __init__(self,
                 binner: Union[str, Binner] = 'quantile',
                 bins: Union[int, List[float]] = 10,
                 min_for_bin: Optional[int] = 10,
                 max_num_cats: Optional[int] = 50,
                 distributor: Union[str, Distributor] = 'fit',
                 seed: Optional[int] = None,
                 ):
        if isinstance(binner, str):
            if isinstance(bins, int):
                self.binner = BINNERS[binner](bins)
            else:
                self.binner = AprioriBinner(bins)
        else:
            self.binner = binner
        self.min_for_bin = min_for_bin
        self.max_num_cats = max_num_cats
        self.distributor = (
            DISTRIBUTORS[distributor](seed=seed)
            if isinstance(distributor, str) else distributor
        )
        self._active = False

    def fit(self, series: pd.Series) -> None:
        transformed = series
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            if self.min_for_bin is None or n_unique >= self.min_for_bin:
                cuts = [series.min()] + self.binner.get(series) + [series.max() + 1]
                if n_unique >= len(frozenset(cuts)):
                    transformed = pd.cut(series, cuts, include_lowest=True)
                    self._active = True
                    self.index = transformed.cat.categories
                    self.distributors = self._fit_distributors(series, transformed)
                    self.dtype = series.dtype
        if self.max_num_cats is not None:
            n_transformed = transformed.nunique()
            if n_transformed > self.max_num_cats:
                raise ValueError(f'too many categories for {series.name} ({n_transformed})')

    def _fit_distributors(self,
                          original: pd.Series,
                          transformed: pd.Series,
                          ) -> List[Distributor]:
        distributors = []
        for cat, bin_vals in original.groupby(transformed):
            d = self.distributor.copy()
            d.fit(bin_vals.values)
            distributors.append(d)
        return distributors

    def transform(self, series: pd.Series) -> pd.Series:
        if self._active:
            return pd.cut(series, self.index)
        else:
            return series

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        if self._active:
            reconstructed = pd.Series(0, dtype=self.dtype, index=series.index)
            for category, distributor in zip(self.index, self.distributors):
                locator = (series == category)
                reconstructed[locator] = distributor.generate(locator.sum())
            return reconstructed.astype(self.dtype)
        else:
            return series

