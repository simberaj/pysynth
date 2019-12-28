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
from typing import Union, Optional, List, Dict, Callable

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection
import sklearn.neighbors


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
        ).drop_duplicates().tolist()


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

    def sample(self, n: int) -> np.ndarray:
        '''Generate a given count of random values from the fitted distribution.'''
        raise NotImplementedError

    @classmethod
    def create(cls, code: str, *args, **kwargs):
        return cls.CODES[code](*args, **kwargs)


class SelectingDistributor:
    '''Randomly sample from a value set according to value frequencies.

    Useful for variables with a small number of unique values.
    '''
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def copy(self) -> SelectingDistributor:
        return SelectingDistributor(seed=self.seed)

    def fit(self, values: np.ndarray) -> SelectingDistributor:
        valcounts = pd.Series(values).value_counts()
        self.targets = valcounts.index.values
        self.probs = valcounts.values.astype(float)
        self.probs /= self.probs.sum()

    def sample(self, n: int) -> np.ndarray:
        return np.random.choice(self.targets, size=n, p=self.probs)


class DiscreteDistributor(Distributor):
    CODES = {
        'select': SelectingDistributor,
    }


class MeanDistributor:
    '''Reproduce the values as a constant value of their mean.'''
    def __init__(self, seed=None):
        pass

    def copy(self) -> MeanDistributor:
        return MeanDistributor()

    def fit(self, values: np.ndarray) -> MeanDistributor:
        self.mean = values.mean()
        return self

    def sample(self, n: int) -> np.ndarray:
        return np.full(n, self.mean)


class StatisticalDistributor:
    '''Reproduce the values from a univariate distribution fitted to the originals.

    Find the continuous distribution from a provided list that
    approximates the distribution of the input values the best according to
    the Kolmogorov-Smirnov two-sample statistic, fit its parameters and sample
    from it. Values outside the range of fitting data are discarded and
    re-sampled.

    :param distributions: `scipy.stats`-like continuous distributions. Need to
        support a class method `fit()` that produces all required constructor
        arguments as a tuple, and a `rvs(int)` method to generate random
        samples. Defaults to `DEFAULT_DISTRIBUTIONS`. The distributions should
        be approximately truncated, otherwise convergence is not guaranteed.
    :param min_samples: Minimum number of generated samples to use when
        evaluating the KS fit statistic.
    :param seed: Random generator seed, applied both before fitting and before
        each generator run.
    '''
    DEFAULT_DISTRIBUTIONS: List[scipy.stats._distn_infrastructure.rv_continuous] = [
        scipy.stats.uniform,
        scipy.stats.truncnorm,
        scipy.stats.truncexpon,
        scipy.stats.triang,
    ]

    def __init__(self,
                 distributions: List[scipy.stats._distn_infrastructure.rv_continuous] = DEFAULT_DISTRIBUTIONS,
                 min_samples: int = 100,
                 seed: Optional[int] = None,
                 ):
        self.distributions = distributions
        self.min_samples = min_samples
        self.seed = seed

    def copy(self) -> StatisticalDistributor:
        return StatisticalDistributor(
            self.distributions,
            self.min_samples,
            self.seed,
        )

    def fit(self, values: np.ndarray) -> StatisticalDistributor:
        self.minval = values.min()
        self.maxval = values.max()
        self.valrange = self.maxval - self.minval
        if self.valrange == 0:
            # does not matter what goes here, will be multiplied by zero anyway
            best_distro = scipy.stats.norm(0, 1)
        else:
            best_distro = None
            normalized = (values - self.minval).astype(float) / self.valrange
            best_fit = 1
            test_size = max(len(normalized), self.min_samples)
            for distro in self.distributions:
                distro_obj = self._fit_distribution(distro, normalized)
                if distro_obj is not None:
                    fit_est = scipy.stats.ks_2samp(
                        normalized,
                        distro_obj.rvs(test_size)
                    )[0]
                    if fit_est < best_fit:
                        best_distro = distro_obj
                        best_fit = fit_est
            if best_distro is None:
                raise ValueError('no distribution could be estimated')
        self.distribution = best_distro
        self.generator = restricted_sampler(
            lambda n: self.minval + self.valrange * self.distribution.rvs(n),
            self.minval,
            self.maxval,
        )
        return self

    def _fit_distribution(self,
                          distribution: scipy.stats._distn_infrastructure.rv_continuous,
                          values: np.ndarray,
                          ) -> Optional[scipy.stats._distn_infrastructure.rv_frozen]:
        try:
            old_setting = np.seterr(all='raise')
            args = distribution.fit(values)
        except FloatingPointError:
            return None
        finally:
            np.seterr(**old_setting)
        if np.isnan(args).any():       # invalid distribution estimated
            return None
        else:
            return distribution(*args)

    def sample(self, n: int) -> np.ndarray:
        np.random.seed(self.seed)
        return self.generator(n)


class KDEDistributor:
    '''Reproduce the values from a kernel density estimate fitted to the originals.

    Find the continuous distribution from a provided list that
    approximates the distribution of the input values the best according to
    the Kolmogorov-Smirnov two-sample statistic, fit its parameters and sample
    from it. Values outside the range of fitting data are discarded and
    re-sampled.

    Warning, this is apparently highly computationally demanding for large
    datasets.

    :param n_bandwidths: Number of tries for the KDE bandwidth estimation,
        in a logarithmic range between .1 and 10. The best fitting output is
        kept using grid search.
    :param seed: Random generator seed, applied both before fitting and before
        each generator run.
    '''
    def __init__(self,
                 n_bandwidths: int = 10,
                 seed: Optional[int] = None,
                 ):
        self.n_bandwidths = n_bandwidths
        self.seed = seed

    def copy(self) -> KDEDistributor:
        return KDEDistributor(
            self.min_unique_continuous,
            self.max_iter,
            self.n_bandwidths,
            self.seed,
        )

    def fit(self, values: np.ndarray) -> KDEDistributor:
        np.random.seed(self.seed)
        bandwidths = 10 ** np.linspace(-1, 1, self.n_bandwidths)
        grid = sklearn.model_selection.GridSearchCV(
            sklearn.neighbors.KernelDensity(),
            {'bandwidth': bandwidths}
        )
        grid.fit(values.reshape(-1, 1))
        self.kde = grid.best_estimator_
        self.generator = restricted_sampler(
            self.kde.sample,
            values.min(),
            values.max(),
        )

    def sample(self, n: int) -> np.ndarray:
        np.random.seed(self.seed)
        return self.generator(n)


def restricted_sampler(generator: Callable[int],
                       minval: float,
                       maxval: float,
                       max_iter: int = 10,
                       ) -> Callable[int]:
    '''Restrict a value generator to a specified range of values.

    :param generator: A function generating random values in specified counts.
    :param minval: Minimum value to generate.
    :param maxval: Maximum value to generate.
    :param max_iter: Maximum number of iterations. If unable to generate enough
        values within range by generating this times more values from the
        underlying generator, fail.
    :raises ValueError: If max_iter is exceeded.
    '''
    def sampler(n):
        g = 0
        i = 0
        results = []
        while g < n:
            if i == max_iter:
                raise ValueError('faulty generator, could not get values in range')
            vals = generator(n)
            sel_vals = vals[(minval <= vals) & (vals <= maxval)]
            results.append(sel_vals)
            g += len(sel_vals)
            i += 1
        return np.hstack(results)[:n]
    return sampler


class ContinuousDistributor(Distributor):
    CODES = {
        'mean': MeanDistributor,
        'statdist': StatisticalDistributor,
        'kde': KDEDistributor,
    }


class SeriesDiscretizer:
    '''Discretize a continuous series to categorical.

    Able to reconstruct variables to their continuous form by estimating
    distributions within bins.

    :param binner: Method to use to determine interval boundaries for
        discretization. Use a :class:`Binner` instance or one of the following
        strings:

        -   `'quantile'` for binning into quantiles (:class:`QuantileBinner`),
        -   `'equalrange'` for binning into equally sized bins
            (:class:`EqualRangeBinner`).

    :param bins: Number of intervals to which to bin non-categorical variables,
        or boundaries of the intervals as a list. If a list is given, it
        overrides the *binner* argument and uses :class:`AprioriBinner`. In
        that case, do not specify the minimum or maximum in the list, just the
        intermediate cuts.
    :param min_unique_continuous: Minimum number of unique values in the input
        to regard a distribution as continuous and not discrete.
    :param discrete_distributor: Method to use to reconstruct numeric values
        for a given category if there is less unique values than
        `min_unique_continuous`. Use a Distributor instance
        or one of the following strings:

        - `'select'` for :class:`SelectingDistributor` (weighted random sampling).

    :param continuous_distributor: Like `discrete_distributor`, but for cases
        when there is many unique values. Use a Distributor instance
        or one of the following strings:

        - `'mean'` for :class:`MeanDistributor` (constant mean value),
        - `'statdist'` for :class:`StatisticalDistributor` (simple estimated distribution),
        - `'kde'` for :class:`KDEDistributor` (KDE-estimated distribution).

    :param seed: Seed for the variable reconstruction.
    '''
    def __init__(self,
                 binner: Union[str, Binner] = 'quantile',
                 bins: Union[int, List[float]] = 10,
                 min_for_bin: Optional[int] = 10,
                 min_unique_continuous: int = 10,
                 discrete_distributor: Union[str, Distributor] = 'select',
                 continuous_distributor: Union[str, Distributor] = 'statdist',
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
        self.min_unique_continuous = min_unique_continuous
        self.discrete_distributor = DiscreteDistributor.create(
            discrete_distributor, seed=seed
        ) if isinstance(discrete_distributor, str) else discrete_distributor
        self.continuous_distributor = ContinuousDistributor.create(
            continuous_distributor, seed=seed
        ) if isinstance(continuous_distributor, str) else continuous_distributor
        self.active = False

    def copy(self) -> SeriesDiscretizer:
        return SeriesDiscretizer(
            binner=self.binner,
            min_for_bin=self.min_for_bin,
            discrete_distributor=self.discrete_distributor,
            continuous_distributor=self.continuous_distributor,
        )

    def fit(self, series: pd.Series) -> SeriesDiscretizer:
        '''Fit the discretizer on a given series.

        Get cut values from the underlying binner, fit distributors for the bins
        and prepare the mapping.

        :raises TypeError: If the series is not numeric.
        '''
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f'cannot discretize a non-numeric series of dtype {series.dtype}')
        n_unique = series.nunique()
        if self.min_for_bin is None or n_unique >= self.min_for_bin:
            cuts = self._get_cuts(series)
            if n_unique >= len(frozenset(cuts)):
                transformed = pd.cut(series, cuts, include_lowest=True)
                self.active = True
                self.index = transformed.cat.categories
                self.distributors = self._fit_distributors(series, transformed)
                self.dtype = series.dtype
        return self

    def _get_cuts(self, series: pd.Series) -> List[float]:
        cuts = self.binner.get(series)
        minval = series.min()
        if cuts[0] != minval:
            cuts.insert(0, minval)
        cuts.append(series.max() + 1)
        return cuts

    def _fit_distributors(self,
                          original: pd.Series,
                          transformed: pd.Series,
                          ) -> List[Distributor]:
        distributors = []
        for cat, bin_vals in original.groupby(transformed):
            if len(bin_vals.index) > 0:
                n_unique = bin_vals.nunique()
                if n_unique < self.min_unique_continuous:
                    d = self.discrete_distributor.copy()
                else:
                    d = self.continuous_distributor.copy()
                d.fit(bin_vals.values)
            else:
                # no values in bin, return a mean-producing distributor
                # at the center of the interval
                d = MeanDistributor(seed=self.continuous_distributor.seed)
                d.fit(np.array([cat.left, cat.right]))
            distributors.append(d)
        return distributors

    def transform(self, series: pd.Series) -> pd.Series:
        '''Discretize the series to a dtype of :class:`pd.Categorical`.'''
        if self.active:
            return pd.cut(series, self.index)
        else:
            return series

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        '''De-discretize the series to a continuous one.

        For each bin, use the fitted distributor to produce continuous values
        to fill the series.
        '''
        if self.active:
            reconstructed = pd.Series(0, dtype=self.dtype, index=series.index)
            for category, distributor in zip(self.index, self.distributors):
                locator = (series == category)
                reconstructed[locator] = distributor.sample(locator.sum())
            na_loc = series.isna()
            if na_loc.any():
                reconstructed[na_loc] = np.nan
            return reconstructed.astype(self.dtype)
        else:
            return series


class DataFrameDiscretizer:
    '''Discretize all continuous columns in a dataframe to categorical.

    Categorical variables are left untouched.

    :param series_discretizer: A discretizer with setup to use for individual
        series. If this is None, any remaining constructor parameters are
        passed to the constructor of :class:`SeriesDiscretizer`.
        If this is a dictionary, the discretizers are applied to the columns
        denoted by the dictionary keys and the remaining columns are not
        discretized.
    :param max_num_cats: Maximum number of categories to accept. High numbers
        of categories make categorical synthesizers unstable. If any of the
        variables has more distinct values than this number after
        categorization, a ValueError is raised.
    '''
    def __init__(self,
                 series_discretizer: Union[
                    None, SeriesDiscretizer, Dict[str, SeriesDiscretizer]
                 ] = None,
                 max_num_cats: Optional[int] = 50,
                 **kwargs
                 ):
        if isinstance(series_discretizer, dict):
            self.discretizers = series_discretizer
            self.pattern = None
        else:
            self.discretizers = None
            if series_discretizer is None:
                self.pattern = SeriesDiscretizer(**kwargs)
            else:
                self.pattern = series_discretizer
        self.max_num_cats = max_num_cats

    def fit(self, dataframe: pd.DataFrame) -> DataFrameDiscretizer:
        '''Fit series discretizers for all non-categorical columns of the dataframe.

        If per-column discretizers were specified, other columns are ignored.

        :raises TypeError: If any column with an explicitly specified
            per-column discretizer in a constructor dict is not numeric.
        '''
        self.fit_transform(dataframe)
        return self

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.discretizers is None:
            self.discretizers = {
                col: self.pattern.copy()
                for col in dataframe.columns
            }
        transformed = dataframe.copy()
        for col in dataframe.columns:
            if col in self.discretizers:
                if pd.api.types.is_numeric_dtype(dataframe[col]):
                    transformed[col] = self.discretizers[col].fit_transform(
                        dataframe[col]
                    )
                else:
                    if self.pattern is None:
                        raise TypeError(f'column {col} is not numeric but explicit discretizer provided')
                    else:
                        del self.discretizers[col]
                if self.max_num_cats is not None:
                    n_after = transformed[col].nunique()
                    if n_after > self.max_num_cats:
                        raise ValueError(f'too many categories for {col} ({n_after})')
        return transformed

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Discretize all non-categorical columns.'''
        dataframe = dataframe.copy()
        for col in self.discretizers:
            dataframe[col] = self.discretizers[col].transform(dataframe[col])
        return dataframe

    def inverse_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Return all formerly non-categorical columns to continuous.'''
        dataframe = dataframe.copy()
        for col in self.discretizers:
            dataframe[col] = self.discretizers[col].inverse_transform(
                dataframe[col]
            )
        return dataframe
