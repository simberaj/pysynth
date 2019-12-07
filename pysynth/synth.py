from typing import Any, Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from . import catdecat


# class Unroller(Protocol):
class Unroller:
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
    :param seed: Random generator seed for the discretizer and unroller.
        (If a custom discretizer is specified, its seed is not overwritten by
        this setting.)
    '''
    def __init__(self,
                 cond_dim: int = 2,
                 categorizer: Optional[catdecat.DataFrameDiscretizer] = None,
                 unroller: Union[str, Unroller] = 'lrem',
                 ignore_cols: List[str] = [],
                 seed: Optional[int] = None,
                 ):
        self.cond_dim = cond_dim
        self.unroller = (
            UNROLLERS[unroller](seed=seed) if isinstance(unroller, str)
            else unroller
        )
        self.discretizer = (
            discretizer if discretizer is not None
            else catdecat.DataFrameDiscretizer(seed=seed)
        )
        self.ignore_cols = ignore_cols

    def fit(self, dataframe: pd.DataFrame) -> None:
        '''Prepare the synthesis according to the provided dataframe.

        :param dataframe: Dataframe to synthesize. Every column is replicated;
            if there are any identifier columns that should not be replicated,
            remove them beforehand.
        '''
        discrete = self.discretize(dataframe.drop(self.ignore_cols, axis=1))
        marginals, axis_values = self._get_marginals(discrete)
        self.axis_values = axis_values
        self.synthed_matrix = ipf(
            self._compute_seed_matrix(discrete),
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
        return self.dediscretize(self._map_axes(self.unroller.unroll(matrix)))

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Fit the synthesizer and synthesize an equal-size dataframe.'''
        self.fit(dataframe)
        return self.generate()

    @staticmethod
    def _get_marginals(dataframe: pd.DataFrame
                       ) -> Tuple[List[np.ndarray], Dict[str, pd.Series]]:
        marginals = []
        maps = {}
        for col in dataframe:
            valcounts = dataframe[col].value_counts(dropna=False, sort=False)
            valcounts = valcounts[valcounts > 0]
            marginals.append(valcounts.values)
            maps[col] = pd.Series(valcounts.index, index=np.arange(len(valcounts)))
        return marginals, maps

    def _map_axes(array: np.ndarray) -> pd.DataFrame:
        # axis_values: Dict[str, Dict[int, Any]]
        raise NotImplementedError

    def _compute_seed_matrix(dataframe: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
    def discretize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Convert all variables to categorical using my discretizer.'''
        raise NotImplementedError
    
    def dediscretize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Reconstruct all non-categorical variables using my discretizer.'''
        raise NotImplementedError
    
    


def ipf(seed_matrix: np.ndarray,
        marginals: List[np.ndarray],
        precision: float = 1e-9
        ) -> np.ndarray:
    '''Perform iterative proportional fitting (IPF).

    :param seed_matrix: Seed matrix, shows a-priori conditional probabilities
        across dimensions.
    :param marginals: Marginal sums for the IPF dimensions. The marginal sums
        of the output matrix will match these.
    :param precision: Terminate IPF when the largest difference of an
        individual cell value between two iterations drops below this
        threshold.
    '''
    matrix = seed_matrix.astype(float)
    n_dim = len(seed_matrix.shape)
    assert n_dim == len(marginals), 'marginal dimensions do not match IPF seed'
    total = marginals[0].sum()
    for i in range(1, len(marginals)):
        if not np.isclose(marginals[i].sum(), total):
            raise ValueError('marginal sum totals do not match')
    # precompute shapes, indices and values for marginal modifiers
    shapes = {}
    other_dims = {}
    for dim_i in range(n_dim):
        shapes[dim_i] = [-1 if i == dim_i else 1 for i in range(n_dim)]
        other_dims[dim_i] = tuple(i for i in range(n_dim) if i != dim_i)
    marginals = [
        np.array(marginal).reshape(shapes[dim_i])
        for dim_i, marginal in enumerate(marginals)
    ]
    # perform IPF
    diff = precision + 1
    while diff > precision:
        previous = matrix
        for dim_i, marginal in enumerate(marginals):
            dim_sums = matrix.sum(axis=other_dims[dim_i]).reshape(shapes[dim_i])
            matrix = matrix / np.where(dim_sums == 0, 1, dim_sums) * marginal
        diff = abs(matrix - previous).max()
    return matrix

