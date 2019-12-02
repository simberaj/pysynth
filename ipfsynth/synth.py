from typing import Any, Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


def synthesize(dataframe: pd.DataFrame,
               n_rows: Optional[int] = None,
               cond_dim: int = 2,
               categories: Union[int, Dict[str, int]] = 10,
               max_num_cats: Optional[int] = 50,
               generation_method: str = 'lrem',
               generator_seed: Optional[int] = None,
               ) -> pd.DataFrame:
    '''Synthesize a dataframe using iterative proportional fitting.
    
    Creates a dataframe that has similar statistical properties to the original
    but does not replicate its rows directly. Preserves univariate
    distributions and covariate distributions to a chosen degree.
    Non-categorical variables are converted to categorical for synthesis
    and then reconstructed using estimated distributions.
    
    :param dataframe: Dataframe to synthesize. Every column is replicated; if
        there are any identifiers that should not be replicated, remove them
        beforehand.
    :param n_rows: Number of rows for the output dataframe. If not given,
        it will match the input dataframe.
    :param cond_dim: Degree to which to match covariate distributions.
        By default, covariates to degree two (two variables' cross-tables)
        will be preserved. If you set this higher than the number of columns in
        the dataframe, the dataframe will be replicated exactly (except for
        the categorization and decategorization of non-categorical variables).
    :param categories: Number of quantiles to which to categorize
        non-categorical variables. Can also be specified per-variable in a
        dictionary. If a dictionary is used, the number of quantiles can also
        be replaced by directly stated cut values.
    :param max_num_cats: Maximum number of categories to accept. High numbers
        of categories make the underlying IPF algorithm unstable. If any
        variable has more distinct values than this number after categorization,
        a ValueError is raised.
    :param generation_method: Method to use to reconstruct the dataset from
        the synthesized IPF matrix:
        
        -   `'lrem'` uses the deterministic largest remainder method (see
            :func:`generate_lrem` for details).
        -   `'random'` uses the non-deterministic random generation method (see
            :func:`generate_random` for details).
    
    :param generator_seed: Seed for the generation method and the
        non-categorical variable reconstruction.
    '''
    categorized, distributions = categorize(dataframe, categories, max_num_cats)
    marginals, axis_values = get_marginals(categorized)
    seed_matrix = compute_seed_matrix(categorized, cond_dim)
    synthed_matrix = ipf(seed_matrix, marginals)
    if n_rows is not None:
        synthed_matrix *= (n_rows / dataframe.shape[0])
    return decategorize(
        map_axes(
            GENERATORS[generation_method](synthed_matrix, generator_seed),
            axis_values
        ),
        distributions
    )


def categorize(dataframe: pd.DataFrame,
               categories: Union[int, Dict[str, Union[int, List[Number]]]] = 10,
               max_num_cats: int = 42,
               ) -> Tuple[pd.DataFrame, Dict[str, List[Distribution]]]:
    raise NotImplementedError


def decategorize(dataframe: pd.DataFrame,
                 distributions: Dict[str, List[Distribution]],
                 ) -> pd.DataFrame:
    raise NotImplementedError


def get_marginals(dataframe: pd.DataFrame
                  ) -> Tuple[List[np.ndarray], Dict[str, Dict[int, Any]]]:
    raise NotImplementedError


def map_axes(dataframe: pd.DataFrame,
             mappings: Dict[str, Dict[int, Any]],
             ) -> pd.DataFrame:
    raise NotImplementedError


def compute_seed_matrix(dataframe: pd.DataFrame,
                        cond_dim: int
                        ) -> np.ndarray:
    raise NotImplementedError


def ipf(seed_matrix: np.ndarray,
        marginals: List[np.ndarray],
        precision: float = 1e-9
        ) -> np.ndarray:
    raise NotImplementedError


def generate_lrem(matrix: np.ndarray,
                  seed: Optional[int] = None
                  ) -> pd.DataFrame:
    raise NotImplementedError


def generate_random(matrix: np.ndarray,
                  seed: Optional[int] = None
                  ) -> pd.DataFrame:
    raise NotImplementedError


GENERATORS = {
    'lrem': generate_lrem,
    'random': generate_random,
}

