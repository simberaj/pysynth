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

SEED_GEN_PARAMS = [
    ((4, 4), 0),
    ((8, 5), 0),
    ((5, 3, 3), 0),
    ((2, 8, 7, 4, 3), 0),
    ((4, 4), 0.1),
    ((8, 5), 0.2),
    ((5, 3, 3), 0.1),
    ((2, 8, 7, 4, 3), 0.05),
]

def generate_seed_matrix(shape, zero_fraction):
    seed_matrix = np.random.rand(*shape)
    if zero_fraction > 0:
        seed_matrix[np.random.rand(*shape) < zero_fraction] = 0
    return seed_matrix

@pytest.mark.parametrize('shape, zero_fraction', SEED_GEN_PARAMS)
def test_ipf_correct(shape, zero_fraction):
    seed_matrix = generate_seed_matrix(shape, zero_fraction)
    marginals = [
        np.random.rand(dim) for dim in shape
    ]
    for i, marginal in enumerate(marginals):
        margsum = marginal.sum()
        marginals[i] = np.array([val * 50 / margsum for val in marginal])
    ipfed = pysynth.ipf.ipf(marginals, seed_matrix, precision=IPF_PRECISION)
    # check the shape and zeros are retained
    assert ipfed.shape == shape
    assert ((seed_matrix == 0) == (ipfed == 0)).all()
    for i, marginal in enumerate(marginals):
        ipfed_sum = ipfed.sum(axis=tuple(j for j in range(ipfed.ndim) if j != i))
        # check the marginal sums match
        assert (abs(ipfed_sum - marginal) < (IPF_PRECISION * ipfed.size / len(marginal))).all()

def test_ipf_dim_mismatch():
    with pytest.raises(ValueError):
        pysynth.ipf.ipf(list(np.ones((3,2))), np.random.rand(2,2))

def test_ipf_sum_mismatch():
    with pytest.raises(ValueError):
        pysynth.ipf.ipf([np.ones(2), np.full(2, 2)], np.random.rand(2,2))

def test_ipf_shape_mismatch():
    with pytest.raises(ValueError):
        pysynth.ipf.ipf([np.ones(2), np.full((2, 4), .25)], np.random.rand(2,2))

@pytest.mark.parametrize('openml_id', [31, 1461, 40536])
def test_get_axis_values(openml_id):
    df = test_data.get_openml(openml_id)
    df = df.drop(
        [col for col, dtype in df.dtypes.iteritems() if not pd.api.types.is_categorical_dtype(dtype)],
        axis=1
    )
    maps = pysynth.ipf.get_axis_values(df)
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

@pytest.mark.parametrize('mat', UNROUND_MATRICES)
def test_lrem_rounder(mat):
    result = pysynth.ipf.LargestRemainderRounder().round(mat)
    assert abs(result - np.round(mat)).max() <= 1

@pytest.mark.parametrize('mat', [
    np.array([[[2,1],[0,0]],[[1,1],[1,3]],[[1,0],[1,2]]]),
    (np.random.rand(4,8,7) * 3).astype(int),
    (np.where(np.random.rand(3,7,4,2) < .2, 0, np.random.rand(3,7,4,2) * 2)).astype(int),
])
def test_unroll(mat):
    unrolled = pysynth.ipf.unroll(mat)
    assert unrolled.shape == (mat.sum(), mat.ndim)
    assert (unrolled >= 0).all()
    for dim_i, dim in enumerate(mat.shape):
        assert (unrolled[:,dim_i] < dim).all()
    unroll_df = pd.DataFrame(unrolled)
    for index, subdf in unroll_df.groupby(unroll_df.columns.tolist()):
        assert mat[index] == len(subdf.index)


def test_map_axes():
    n_cols = 6
    n_cats = 5
    indices = (np.random.rand(40, n_cols) * n_cats).astype(int)
    axis_values = {
        chr(97 + np.random.randint(26)): pd.Series(
            [chr(97 + k) for k in np.random.randint(26, size=n_cats)],
            index=np.arange(n_cats)
        ) for i in range(n_cols)
    }
    df = pysynth.ipf.map_axes(indices, axis_values)
    assert list(df.columns) == list(axis_values.keys())
    assert len(df.index) == indices.shape[0]
    i = 0
    for col, mapping in axis_values.items():
        for index, value in mapping.iteritems():
            assert (df[col][indices[:,i] == index] == value).all()
        i += 1

def test_calc_true_matrix():
    ipfsynth = pysynth.ipf.IPFSynthesizer(cond_dim=2)
    n_rows = 400
    n_cols = 4
    n_cats = 5
    cat_indices = np.arange(n_cats)
    cat_objs = np.array([chr(97 + k) for k in np.arange(n_cats)])
    map_to_objs = pd.Series(cat_objs, index=cat_indices)
    ind_df = pd.DataFrame(
        np.random.randint(n_cats, size=(n_rows, n_cols)),
        columns=[chr(97 + k) for k in np.arange(n_cols)]
    )
    cat_df = ind_df.copy()
    for col in cat_df.columns:
        cat_df[col] = cat_df[col].map(map_to_objs)
    ipfsynth.axis_values = {
        col: map_to_objs for col in cat_df.columns
    }
    true_mat = ipfsynth.calc_true_matrix(cat_df)
    assert true_mat.shape == tuple([n_cats] * n_cols)
    assert true_mat.min() >= 0
    assert true_mat.sum() == n_rows
    assert np.isclose(true_mat, true_mat.astype(int)).all()
    for inds, subdf in ind_df.groupby(list(ind_df.columns)):
        assert true_mat[inds] == len(subdf.index)

# @pytest.mark.parametrize('shape, zero_fraction', [((4, 3, 2), .1)])
@pytest.mark.parametrize('shape, zero_fraction', SEED_GEN_PARAMS)
def test_obscure_seed(shape, zero_fraction):
    seed_matrix = (generate_seed_matrix(shape, zero_fraction) * 10).astype(int)
    n_dims = seed_matrix.ndim
    for cond_dim in range(1, min(n_dims + 1, 4)):
        obscured = pysynth.ipf.obscure_seed(seed_matrix, cond_dim)
        preserved_dims = [
            tuple(sorted(frozenset(dims)))
            for dims in itertools.combinations_with_replacement(
                range(n_dims), cond_dim
            )
        ]
        print(preserved_dims)
        for sel_dim_is in preserved_dims:
            other_dim_is = tuple(i for i in range(n_dims) if i not in sel_dim_is)
            assert np.isclose(
                obscured.sum(axis=other_dim_is),
                seed_matrix.sum(axis=other_dim_is)
            ).all()

# @pytest.mark.parametrize('openml_id', [11, 31, 1461, 1480])
@pytest.mark.parametrize('openml_id, rder', [(11, 'lrem'), (23, 'random')])
def test_synth(openml_id, rder):
    df = test_data.get_openml(openml_id)
    synth = pysynth.ipf.IPFSynthesizer(rounder=rder).fit_transform(df)
    assert frozenset(df.columns) == frozenset(synth.columns)
    assert len(df.index) == len(synth.index)
    for col in df.columns:
        assert df[col].dtype == synth[col].dtype
        if pd.api.types.is_categorical_dtype(df[col].dtype):
            assert frozenset(synth[col].cat.categories).issubset(df[col].cat.categories)
        else:
            assert df[col].min() <= synth[col].mean() <= df[col].max()
