from typing import Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import itertools

from . import catdecat


class MatrixRounder:
    def round(self, matrix: np.ndarray) -> np.ndarray:
        '''Round a matrix to integers, preserving its grand total.'''
        raise NotImplementedError


class LargestRemainderRounder(MatrixRounder):
    '''Round a matrix to integers using the largest-remainder method.

    The largest-remainder method (Hare quota) is deterministic and allocates
    roundings to the largest remainders. Ties are broken by selecting the cells
    with largest indices.

    :param seed: Meaningless, this method is deterministic.
    '''
    def __init__(self, seed: Optional[int] = None):
        pass    # this is a deterministic rounder

    def round(self, matrix: np.ndarray) -> np.ndarray:
        # round down to integers, those are sure hits
        rounded = matrix.astype(int)
        # compute remainders to be distributed
        remainders = matrix - rounded
        sum_remaining = int(np.round(remainders.sum()))
        # locate sum_remaining largest remainders
        ind_add = np.argsort(
            remainders, axis=None, kind='stable'
        )[::-1][:sum_remaining]
        rounded[np.unravel_index(ind_add, matrix.shape)] += 1
        return rounded


class RandomSamplingRounder(MatrixRounder):
    '''Round a matrix to integers using random sampling.

    Randomly sample from matrix cells, using their values as probabilities,
    until the sum is matched.

    :param seed: Seed for the random sampler.
    '''
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def round(self, matrix: np.ndarray) -> np.ndarray:
        matrix_sum = matrix.sum()
        final_total = int(np.round(matrix_sum))
        probs = (matrix / matrix_sum).flatten()
        # print('PROBS', probs.sum())
        np.random.seed(self.seed)
        # randomly select cells to be included
        bucket_is = np.random.choice(len(probs), size=final_total, p=probs)
        # count the cells
        cell_counts = np.bincount(bucket_is)
        return np.hstack((
            cell_counts,
            np.zeros(matrix.size - len(cell_counts), dtype=cell_counts.dtype)
        )).reshape(*matrix.shape)


ROUNDERS = {
    'lrem': LargestRemainderRounder,
    'random': RandomSamplingRounder,
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
    :param discretizer: A :class:`catdecat.DataFrameDiscretizer` instance to
        convert numeric variables to and from categorical ones.
        Can be specified as a single instance or per variable in a dictionary.
        If not given, a single instance with default setup will be created.
    :param rounder: Method to use to round the IPF matrix to integer counts to
        enable row generation. Use a MatrixRounder instance or one of the
        following strings:

        -   `'lrem'` uses the deterministic largest remainder method (see
            :class:`LargestRemainderRounder` for details) which is more suited
            to small datasets.
        -   `'random'` uses the non-deterministic random generation method (see
            :class:`RandomSamplingRounder` for details), more suited to larger
            datasets.

    :param ignore_cols: Columns from the input dataframe to not synthesize
        (identifiers etc.); will be omitted from the output.
    :param seed: Random generator seed for the discretizer and unroller.
        (If a custom discretizer is specified, its seed is not overwritten by
        this setting.)
    '''
    def __init__(self,
                 cond_dim: int = 2,
                 discretizer: Optional[catdecat.DataFrameDiscretizer] = None,
                 rounder: Union[str, MatrixRounder] = 'lrem',
                 ignore_cols: List[str] = [],
                 seed: Optional[int] = None,
                 ):
        if cond_dim < 1:
            raise ValueError('cannot preserve less than one-dimensional sums')
        self.cond_dim = cond_dim
        self.rounder = (
            ROUNDERS[rounder](seed=seed) if isinstance(rounder, str)
            else rounder
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
        discrete = self.discretizer.fit_transform(
            dataframe.drop(self.ignore_cols, axis=1)
        )
        # marginals, axis_values = get_marginals(discrete)
        self.axis_values = get_axis_values(discrete)
        self.synthed_matrix = obscure_seed(
            self.calc_true_matrix(discrete), self.cond_dim
        )
        self.original_n_rows = dataframe.shape[0]

    def sample(self, n: Optional[int] = None) -> pd.DataFrame:
        '''Generate a synthetic dataframe with a given number of rows.

        :param n: Number of rows for the output dataframe. If not given,
            it will match the fitting dataframe.
        '''
        matrix = self.synthed_matrix
        if n is not None:
            matrix *= (n / self.original_n_rows)
        return self.discretizer.inverse_transform(
            map_axes(unroll(self.rounder.round(matrix)), self.axis_values)
        )

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Fit the synthesizer and synthesize an equal-size dataframe.'''
        self.fit(dataframe)
        return self.sample()

    def calc_true_matrix(self, dataframe: pd.DataFrame) -> np.ndarray:
        '''Calculate a IPF matrix reflecting true observation frequencies.'''
        for col, mapper in self.axis_values.items():
            dataframe[col] = dataframe[col].map(
                pd.Series(mapper.index, index=mapper.values)
            )
        true_seed = np.zeros(tuple(len(mapper) for mapper in self.axis_values.values()))
        for indices in dataframe.itertuples(index=False, name=None):
            true_seed[indices] += 1
        return true_seed


def ipf(marginals: List[np.ndarray],
        seed_matrix: Optional[np.ndarray] = None,
        precision: float = 1e-9
        ) -> np.ndarray:
    '''Perform iterative proportional fitting (IPF) on 1D marginal sums.

    Reformats the marginals to a generic n-D format and then delegates to
    :func:`ipf_multidim`.

    :param marginals: Marginal sums for the IPF dimensions. The marginal sums
        of the output matrix will match these. The list should contain
        one-dimensional arrays that sum to the same number.
    :param seed_matrix: Seed matrix, shows a-priori conditional probabilities
        across dimensions.
    :param precision: Terminate IPF when the largest difference of an
        individual cell value between two iterations drops below this
        threshold.
    '''
    n_dim = len(marginals)
    if seed_matrix is not None and len(seed_matrix.shape) != n_dim:
        raise ValueError('marginal dimensions do not match IPF seed')
    return ipf_multidim(
        [
            marginal.reshape([
                -1 if i == dim_i else 1 for i in range(n_dim)
            ])
            for dim_i, marginal in enumerate(marginals)
        ],
        seed_matrix,
        precision=precision
    )


def ipf_multidim(marginals: List[np.ndarray],
                 seed_matrix: Optional[np.ndarray] = None,
                 precision: float = 1e-9
                 ) -> np.ndarray:
    '''Perform iterative proportional fitting (IPF) on arbitrary marginal sums.

    :param marginals: Marginal sums for the final matrix. The list should
        contain arrays with equal sums. Their dimensions should correspond to
        the seed matrix or be 1 - at dimensions for which the given marginal
        sum is summed (contracted).
    :param seed_matrix: Seed matrix, shows a-priori conditional probabilities
        across dimensions. If not given, the matrix shape will be computed from
        the marginals and it will be initialized by ones.
    :param precision: Terminate IPF when the largest difference of an
        individual cell value between two iterations drops below this
        threshold.
    '''
    if seed_matrix is None:
        shape = tuple(
            max(marg.shape[i] for marg in marginals)
            for i in range(min(marg.ndim for marg in marginals))
        )
        matrix = np.ones(shape)
    else:
        matrix = seed_matrix.astype(float)
        shape = matrix.shape
    ipf_check_marginals(marginals, shape)
    other_dims = [
        tuple(
            dim_i for dim_i in range(len(shape))
            if marginal.shape[dim_i] == 1
        )
        for marginal in marginals
    ]
    diff = precision + 1
    while diff > precision:
        previous = matrix
        for marginal, other_dimtup in zip(marginals, other_dims):
            dim_sums = matrix.sum(axis=other_dimtup).reshape(marginal.shape)
            matrix = matrix / np.where(dim_sums == 0, 1, dim_sums) * marginal
        diff = abs(matrix - previous).max()
    return matrix


def ipf_check_marginals(marginals: List[np.ndarray], shape: Tuple[int]) -> None:
    '''Checks whether the marginal sums are valid for IPF of given shape.

    Used internally by :func:`ipf_multidim` so uses the format of marginals
    required by that function.

    :param marginals: List of marginal sum arrays to be checked.
    :param shape: Shape of the resulting matrix.
    '''
    total = marginals[0].sum()
    for i, marginal in enumerate(marginals):
        if i != 0 and not np.isclose(marginal.sum(), total):
            raise ValueError('marginal sum totals do not match')
        if marginal.ndim != len(shape):
            raise ValueError('marginal dimensions do not match seed')
        for j, mshape in enumerate(marginal.shape):
            if mshape != 1 and mshape != shape[j]:
                raise ValueError('marginal shape does not match seed')


def unroll(matrix: np.ndarray) -> np.ndarray:
    '''Convert a matrix of cell counts to a matrix of cell indices with those counts.

    :param matrix: A matrix of non-negative integers denoting counts of
        observations. Each cell will generate this many rows with its positional
        indices.
    '''
    cumcounts = np.cumsum(matrix)
    inds = np.zeros(cumcounts[-1], dtype=int)
    np.add.at(inds, cumcounts[:np.searchsorted(cumcounts, cumcounts[-1])], 1)
    return np.stack(np.unravel_index(
        np.cumsum(inds), matrix.shape
    )).transpose()


def map_axes(indices: np.ndarray,
             axis_values: Dict[str, pd.Series],
             ) -> pd.DataFrame:
    '''Convert a category index array to a dataframe with categories.

    :param indices: A 2-D integer array.
    :param axis_values: A dictionary with length matching the column count of
        `indices`. Its keys are names of the columns to be assigned to the
        dataframe, while values map the category indices from the given column
        of the integer array to the expected dataframe values.
    '''
    dataframe = pd.DataFrame(indices, columns=list(axis_values.keys()))
    for col, mapper in axis_values.items():
        dataframe[col] = dataframe[col].map(mapper)
    return dataframe


def obscure_seed(true: np.ndarray,
                 cond_dim: int = 2
                 ) -> np.ndarray:
    '''Produce a matrix preserving some cross-sums of the original.

    :param true: The matrix to be obscured. The output matrix will match
        sums of its cells as aggregated to each combination of `cond_dim`
        dimensions.
    :param cond_dim: The number of dimensions to preserve cross-sums for.
    '''
    if cond_dim < 1:
        raise ValueError('invalid preservation dimension count')
    marginals = []
    dim_is = list(range(true.ndim))
    for sel_dim_is in itertools.combinations(dim_is, cond_dim):
        left_dim_is = []
        sum_indexer = []
        for dim_i in dim_is:
            if dim_i in sel_dim_is:
                sum_indexer.append(true.shape[dim_i])
            else:
                sum_indexer.append(1)
                left_dim_is.append(dim_i)
        marginals.append(true.sum(axis=tuple(left_dim_is)).reshape(sum_indexer))
    return ipf_multidim(marginals)


def get_axis_values(dataframe: pd.DataFrame
                    ) -> Dict[str, pd.Series]:
    '''Compute mappings of indices to categories for each dataframe column.'''
    maps = {}
    for col in dataframe:
        values = pd.Series(dataframe[col].unique()).sort_values().values
        maps[col] = pd.Series(values, index=np.arange(len(values)))
    return maps
