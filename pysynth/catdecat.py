
from __future__ import annotations
# from typing import Union, Optional, List, Dict, Tuple, Protocol
from typing import Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats


# class Binner(Protocol):
class Binner:
    '''Interface for numeric variable interval boundary determiners.'''
    def get(self, data: pd.Series) -> List[float]:
        '''Return a list of numbers denoting right-inclusive cut values.'''
        raise NotImplementedError


class QuantileBinner(Binner):
    '''A binner that gives quantile cuts.'''
    def __init__(self, bins: int):
        self.bins = bins
    
    def get(self, data: pd.Series) -> List[float]:
        return data.quantile(
            (np.arange(self.bins - 1) + 1) / self.bins
        ).tolist()


class EqualRangeBinner(Binner):
    '''A binner that gives equal-range cuts.'''
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
    def __init__(self, seed = None):
        pass
    
    def copy(self) -> MeanDistributor:
        return MeanDistributor()

    def fit(self, values: np.ndarray) -> None:
        '''Record mean of the values to reproduce as a constant later.'''
        self.mean = values.mean()
    
    def generate(self, n: int) -> np.ndarray:
        return np.full(n, self.mean)


class FittingDistributor:
    DEFAULT_DISTROS = [
        scipy.stats.uniform,
        scipy.stats.truncnorm,
        scipy.stats.truncexpon,
        scipy.stats.triang,
        scipy.stats.trapz,
    ]

    def __init__(self,
                 distributions: List[scipy.stats._distn_infrastructure.rv_continuous] = DEFAULT_DISTROS,
                 min_samples: int = 100,
                 seed: Optional[int] = None,
                 ):
        self.distributions = distributions
        self.min_samples = min_samples
        self.seed = seed
    
    def copy(self) -> FittingDistributor:
        return FittingDistributor(
            self.distributions, self.min_samples, self.seed
        )
        
    def fit(self, values: np.ndarray) -> None:
        np.random.seed(self.seed)
        self.min = values.min()
        self.range = values.max() - self.min
        normalized = (values - self.min) / self.range
        best_fit = 1
        for distro in self.distributions:
            distro_obj = distro(*distro.fit(normalized))
            fit_est = scipy.stats.ks_2samp(
                normalized,
                distro_obj.rvs(max(len(normalized), self.min_samples))
            )[0]
            if fit_est < best_fit:
                self.distribution = distro_obj
                best_fit = fit_est
    
    def generate(self, n: int) -> np.ndarray:
        np.random.seed(self.seed)
        return self.min + self.range * self.distribution.rvs(n)
            


DISTRIBUTORS = {
    'mean': MeanDistributor,
    'fit': FittingDistributor,
}


class Categorizer:
    '''Converts continuous variables to categorical.
    
    Able to reconstruct variables to their continuous form by estimating
    distributions within bins. Categorical variables are passed through
    unchanged.
    
    :param binner: Method to use to determine interval boundaries for
        conversion of numeric variables to categorical. Use a Binner instance
        or one of the following strings:
        
        -   `'quantile'` for binning into quantiles (:class:`QuantileBinner`),
        -   `'equalrange'` for binning into equally sized bins
            (:class:`EqualRangeBinner`).
        
    :param bins: Number of intervals to which to bin non-categorical variables,
        or boundaries of the intervals as a list. If a list is given, do not
        specify the minimum or maximum, just the intermediate cuts.
    :param max_num_cats: Maximum number of categories to accept. High numbers
        of categories make the synthesizer unstable. If the
        variable has more distinct values than this number after
        categorization, a ValueError is raised.
    :param distributor: Method to use to reconstruct numeric values for a given
        category. Use a Distributor instance
        or one of the following strings:
        
        - `'mean'` for :class:`MeanDistributor` (constant mean value)
        
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
        self.seed = seed
        self.active = False

    def fit(self, series: pd.Series) -> None:
        transformed = series
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            if self.min_for_bin is None or n_unique >= self.min_for_bin:
                cuts = [series.min()] + self.binner.get(series) + [series.max() + 1]
                if n_unique >= len(frozenset(cuts)):
                    transformed = pd.cut(series, cuts, include_lowest=True)
                    self.active = True
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
            d.fit(bin_vals)
            distributors.append(d)
        return distributors

    def transform(self, series: pd.Series) -> pd.Series:
        if self.active:
            return pd.cut(series, self.index)
        else:
            return series

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        if self.active:
            reconstructed = pd.Series(0, dtype=self.dtype, index=series.index)
            for category, distributor in zip(self.index, self.distributors):
                locator = (series == category)
                reconstructed[locator] = distributor.generate(locator.sum())
            return reconstructed.astype(self.dtype)
        else:
            return series
            



