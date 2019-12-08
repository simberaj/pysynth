import sys
import os
import itertools

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth.ipf
import test_data

IPF_PRECISION = 1e-10

np.random.seed(1711)

@pytest.mark.parametrize('shape, zero_fraction', [
        ((4, 4), 0),
        ((8, 5), 0),
        ((5, 3, 3), 0),
        ((2, 8, 7, 4, 3), 0),
        ((4, 4), 0.1),
        ((8, 5), 0.3),
        ((5, 3, 3), 0.1),
        ((2, 8, 7, 4, 3), 0.05),
    ]
)
def test_ipf_correct(shape, zero_fraction):
    seed_matrix = np.random.rand(*shape)
    if zero_fraction > 0:
        seed_matrix[np.random.rand(*shape) < zero_fraction] = 0
    marginals = [
        np.random.rand(dim) for dim in shape
    ]
    for i, marginal in enumerate(marginals):
        margsum = marginal.sum()
        marginals[i] = np.array([val * 50 / margsum for val in marginal])
    ipfed = pysynth.ipf.ipf(seed_matrix, marginals, precision=IPF_PRECISION)
    # check the shape and zeros are retained
    assert ipfed.shape == shape
    assert ((seed_matrix == 0) == (ipfed == 0)).all()
    for i, marginal in enumerate(marginals):
        ipfed_sum = ipfed.sum(axis=tuple(j for j in range(ipfed.ndim) if j != i))
        # check the marginal sums match
        assert (abs(ipfed_sum - marginal) < (IPF_PRECISION * ipfed.size / len(marginal))).all()

def test_ipf_dim_mismatch():
    with pytest.raises(ValueError):
        pysynth.ipf.ipf(np.random.rand(2,2), list(np.ones((3,2))))

def test_ipf_sum_mismatch():
    with pytest.raises(ValueError):
        pysynth.ipf.ipf(np.random.rand(2,2), [np.ones(2), np.full(2, 2)])

@pytest.mark.parametrize('openml_id', [31, 1461, 40536])
def test_get_marginals(openml_id):
    df = test_data.get_openml(openml_id)
    df = df.drop(
        [col for col, dtype in df.dtypes.iteritems() if not pd.api.types.is_categorical_dtype(dtype)],
        axis=1
    )
    margs, maps = pysynth.ipf.IPFSynthesizer._get_marginals(df)
    for i, marg in enumerate(margs):
        assert np.issubdtype(marg.dtype, np.integer)
        assert len(marg) == df[df.columns[i]].nunique(dropna=False)
        assert (marg >= 0).all()
        assert marg.sum() == len(df.index)
    for col in maps:
        assert col in df.columns
        assert (maps[col].index == np.arange(len(maps[col].index))).all()
        assert frozenset(maps[col].values) == frozenset(df[col].unique())

ROUNDERS = [
    pysynth.ipf.LargestRemainderRounder(),
    pysynth.ipf.RandomSamplingRounder(seed=1711),
]

UNROUND_MATRICES = [
    np.array([[[2.,.5],[.5,0]],[[1.2,1],[1,.8]],[[1,.2],[1,1.8]]]),
    np.random.rand(4,8,7) * 3,
    np.where(np.random.rand(3,7,4,2) < .2, 0, np.random.rand(3,7,4,2) * 2),
]

@pytest.mark.parametrize('rder, mat', list(itertools.product(
    ROUNDERS, UNROUND_MATRICES
)))
def test_rounders(rder, mat):
    result = rder.round(mat)
    assert np.issubdtype(result.dtype, np.integer)
    assert result.sum() == int(np.round(mat.sum()))
    assert result.min() >= 0
    assert result[mat == 0].sum() == 0
    # for dim_i, dim in enumerate(mat.shape):
        # assert (result[:,dim_i] < dim).all()
