from typing import Any, Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from . import catdecat


class IPFSynthesizer:
    '''Synthesize a dataframe using iterative proportional fitting.
    
    Creates a dataframe that has similar statistical properties to the original
    but does not replicate its rows directly. Preserves univariate
    distributions and covariate distributions to a chosen degree.
    Non-categorical variables are converted to categorical for synthesis
    and then reconstructed using estimated distributions.
    
    :param cond_dim: Degree to which to match covariate distributions.
        By default, covariates to degree two (two variables' cross-tables)
        will be preserved. If you set this higher than the number of columns in
        the dataframe, the dataframe will be replicated exactly (except for
        the categorization and decategorization of non-categorical variables).
    :param categorizer: A :class:`catdecat.Categorizer` instance to convert
        numeric variables to and from categorical ones. Can be specified as a
        single instance or per variable in a dictionary. If not given, a single
        instance with default setup will be created.
    :param unroller: Method to use to reconstruct the dataset from
        the synthesized IPF matrix. Use a Unroller instance or one of the
        following strings:
        
        -   `'lrem'` uses the deterministic largest remainder method (see
            :func:`generate_lrem` for details).
        -   `'random'` uses the non-deterministic random generation method (see
            :func:`generate_random` for details).
    
    :param ignore_cols: Columns from the input dataframe to not synthesize
        (identifiers etc.); will be omitted from the output.
    :param seed: Random generator seed for the categorizer and unroller.
        (If a custom categorizer is specified, its seed is not overwritten by
        this setting.)
    '''
    def __init__(self,
                 cond_dim: int = 2,
                 categorizer: Union[
                     None,
                     catdecat.Categorizer,
                     Dict[str, catdecat.Categorizer]
                 ] = None,
                 unroller: Union[str, Unroller] = 'lrem',
                 ignore_cols: List[str] = [],
                 seed: Optional[int] = None,
                 ):
        self.cond_dim = cond_dim
        self.unroller = (
            UNROLLERS[unroller](seed=seed) if isinstance(unroller, str)
            else unroller
        )
        self.categorizer = (
            categorizer if categorizer is not None
            else catdecat.Categorizer(seed=seed)
        )
        self.ignore_cols = ignore_cols
    
    def fit(self, dataframe: pd.DataFrame) -> None:
        '''Prepare the synthesis according to the provided dataframe.
        
        :param dataframe: Dataframe to synthesize. Every column is replicated;
            if there are any identifier columns that should not be replicated,
            remove them beforehand.
        '''
        categorized = self.categorize(dataframe.drop(self.ignore_cols, axis=1))
        marginals, axis_values = self._get_marginals(categorized)
        self.axis_values = axis_values
        self.synthed_matrix = ipf(
            self._compute_seed_matrix(categorized),
            marginals
        )
        self.original_n_rows = dataframe.shape[0]
    
    def generate(self, n_rows: Optional[int]) -> pd.DataFrame:
        '''Generate a synthetic dataframe with a given number of rows.
        
        :param n_rows: Number of rows for the output dataframe. If not given,
            it will match the fitting dataframe.
        '''
        matrix = self.synthed_matrix
        if n_rows is not None:
            matrix *= (n_rows / self.original_n_rows)
        return self.decategorize(self._map_axes(self.unroller.unroll(matrix)))
    
    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Fit the synthesizer and synthesize an equal-size dataframe.'''
        self.fit(dataframe)
        return self.generate()
    
    def _get_marginals(dataframe: pd.DataFrame
                       ) -> Tuple[List[np.ndarray], Dict[str, Dict[int, Any]]]:
        raise NotImplementedError
        
    def _map_axes(array: np.ndarray) -> pd.DataFrame:
        # axis_values: Dict[str, Dict[int, Any]]
        raise NotImplementedError

    def _compute_seed_matrix(dataframe: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def ipf(seed_matrix: np.ndarray,
        marginals: List[np.ndarray],
        precision: float = 1e-9
        ) -> np.ndarray:
    raise NotImplementedError


class Unroller(Protocol):
    def unroll(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LargestRemainderUnroller(Unroller):
    def unroll(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomSamplerUnroller(Unroller):
    def unroll(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


GENERATORS = {
    'lrem': LargestRemainderUnroller,
    'random': RandomSamplerUnroller,
}

