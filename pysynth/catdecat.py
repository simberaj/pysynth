
from typing import Union, Optional, List, Dict, Tuple, Number, Protocol

import pandas as pd


class Cutter(Protocol):
    '''Interface for numeric variable interval boundary determiners.'''
    def get(self, data: pd.Series) -> List[Number]:
        '''Return a list of numbers denoting right-inclusive cut values.'''
        raise NotImplementedError


CUTTERS = {
}


class Distributor(Protocol):
    '''Interface for numeric variable reconstructors.
    
    Fits itself on values for a single interval, and reproduces the
    distribution for a given number of output values by random sampling.
    '''
    def fit(self, values: np.ndarray) -> None:
        '''Fit a distribution on the values for a given interval.'''
        raise NotImplementedError
    
    def generate(self, n: int) -> np.ndarray:
        '''Generate a given count of random values from the fitted distribution.'''
        raise NotImplementedError


DISTRIBUTORS = {
}


class Categorizer:
    '''Converts continuous variables to categorical.
    
    Able to reconstruct variables to their continuous form by estimating
    distributions within bins. Categorical variables are passed through
    unchanged.
    
    :param cutter: Method to use to determine interval boundaries for
        conversion of numeric variables to categorical. Use a Cutter instance
        or one of the following strings:
        
        - `'todo'`
        
    :param cuts: Number of intervals to which to categorize
        non-categorical variables, or boundaries of the intervals as a list.
    :param max_num_cats: Maximum number of categories to accept. High numbers
        of categories make the synthesizer unstable. If the
        variable has more distinct values than this number after
        categorization, a ValueError is raised.
    :param distributor: Method to use to reconstruct numeric values for a given
        category. Use a Distributor instance
        or one of the following strings:
        
        - `'todo'`
        
    :param seed: Seed for the variable reconstruction.
    '''
    def __init__(self,
                 cutter: Union[str, Cutter] = 'quantile',
                 cuts: Union[int, List[Number]] = 10,
                 max_num_cats: Optional[int] = 50,
                 distributor: Union[str, Distributor] = 'fit',
                 seed: Optional[int] = None,
                 ):
        self.cutter = CUTTERS[cutter] if isinstance(cutter, str) else cutter
        self.categories = categories
        self.max_num_cats = max_num_cats
        self.distributor = (
            DISTRIBUTORS[distributor](seed=seed)
            if isinstance(distributor, str) else distributor
        )
        self.seed = seed

    def fit(self, series: pd.Series) -> None:
        raise NotImplementedError

    def transform(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError



