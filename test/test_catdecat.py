import sys
import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth.catdecat

import test_data

np.random.seed(1711)

@pytest.mark.parametrize('binner_cls, bins', list(itertools.product(
    pysynth.catdecat.BINNERS.values(), [5, 10, 20],
)))
def test_binners_formal(binner_cls, bins):
    binner = binner_cls(bins)
    cutvals = binner.get(pd.Series(np.random.rand(100)))
    assert isinstance(cutvals, list)
    assert len(cutvals) == bins - 1
    assert all(isinstance(cutval, float) for cutval in cutvals)

@pytest.mark.parametrize('bins', [4, 8, 12])
def test_quantile_binner(bins):
    binner = pysynth.catdecat.QuantileBinner(bins)
    for i in range(10):
        vals = pd.Series(np.random.rand(100))
        cuts = binner.get(vals)
        assert np.isclose(
            cuts,
            np.percentile(vals, (np.arange(bins - 1) + 1) / bins * 100)
        ).all()

@pytest.mark.parametrize('bins', [4, 8, 12])
def test_equalrange_binner(bins):
    binner = pysynth.catdecat.EqualRangeBinner(bins)
    for i in range(10):
        vals = pd.Series(np.random.rand(100))
        cuts = binner.get(vals)
        inner_widths = np.diff(cuts)
        assert np.isclose(inner_widths.min(), inner_widths.max())
        assert np.isclose(inner_widths.mean(), cuts[0] - vals.min())
        assert np.isclose(inner_widths.mean(), vals.max() - cuts[-1])

def test_apriori_binner():
    for i in range(10):
        vals = pd.Series(np.random.rand(100))
        cuts = np.sort(vals.sample(10).unique()).tolist()
        binner = pysynth.catdecat.AprioriBinner(cuts)
        assert binner.get(vals) == cuts


@pytest.mark.parametrize('dist_cls', pysynth.catdecat.ContinuousDistributor.CODES.values())
def test_continuous_distributors(dist_cls):
    distributor = dist_cls(seed=42)
    minval = 2
    maxval = 7
    for i in range(10):
        vals = np.random.rand(100) * (maxval - minval) + minval
        distributor.fit(vals)
        reconst = distributor.sample(100)
        assert minval <= reconst.min() <= reconst.max() <= maxval

@pytest.mark.parametrize('dist_cls', pysynth.catdecat.DiscreteDistributor.CODES.values())
def test_discrete_distributors(dist_cls):
    distributor = dist_cls(seed=42)
    minval = 2
    maxval = 12
    for i in range(10):
        vals = (np.random.rand(100) * (maxval - minval) + minval).astype(int)
        uniques = np.unique(vals)
        distributor.fit(vals)
        reconst = distributor.sample(100)
        assert minval <= reconst.min() <= reconst.max() <= maxval
        assert np.isin(reconst, uniques).all()

def test_restricted_sampler_ok():
    minval = 1
    maxval = 3
    testdist = scipy.stats.norm(2, 1)
    sampler = pysynth.catdecat.restricted_sampler(testdist.rvs, minval, maxval)
    x = sampler(1000)
    assert (x >= minval).all()
    assert (x <= maxval).all()
    assert len(x) == 1000

def test_restricted_sampler_fail():
    minval = 1
    maxval = 3
    testgen = lambda n: np.full(n, 4)
    sampler = pysynth.catdecat.restricted_sampler(testgen, 1, 3)
    with pytest.raises(ValueError):
        x = sampler(1000)


def test_mean_distributor():
    dist = pysynth.catdecat.MeanDistributor()
    for i in range(10):
        vals = np.random.rand(100)
        val_mean = vals.mean()
        dist.fit(vals)
        assert (dist.sample(20) == np.array([val_mean] * 20)).all()


SERIES_DISCRETIZERS = [
    pysynth.catdecat.SeriesDiscretizer(seed=42),
    pysynth.catdecat.SeriesDiscretizer(binner='equalrange', continuous_distributor='mean', seed=42),
]

@pytest.mark.parametrize('categ, na_frac', list(itertools.product(
    SERIES_DISCRETIZERS, [0, 0.2, 1]
)))
def test_discretizer_numeric(categ, na_frac):
    size = 100
    minval = -3
    maxval = 10
    vals = pd.Series(np.random.rand(size) * 13 - 3)
    vals[np.random.rand(size) < na_frac] = np.nan
    cats = categ.fit_transform(vals)
    check_series_properly_discretized(vals, cats, categ.inverse_transform(cats))

@pytest.mark.parametrize('n_cats', [2, 20, 70])
def test_discretizer_category(n_cats):
    vals = pd.Series(np.random.choice([chr(48 + i) for i in range(n_cats)], 300))
    c = pysynth.catdecat.SeriesDiscretizer(seed=42)
    with pytest.raises(TypeError):
        trans = c.fit_transform(vals)


@pytest.mark.parametrize('n_vals', [2, 20, 70])
def test_discretizer_integer(n_vals):
    vals = pd.Series(np.random.randint(n_vals, size=300))
    c = pysynth.catdecat.SeriesDiscretizer(seed=42)
    cats = c.fit_transform(vals)
    if n_vals < c.min_for_bin:
        assert (cats == vals).all()
    else:
        check_series_properly_discretized(vals, cats, c.inverse_transform(cats))


def check_df_properly_discretized(df, tr_df, reconst_df, max_nums=10):
    orig_cols = frozenset(df.columns)
    assert orig_cols == frozenset(tr_df.columns)
    assert orig_cols == frozenset(reconst_df.columns)
    for col in df.columns:
        check_series_properly_discretized(
            df[col],
            tr_df[col],
            reconst_df[col],
            max_nums=max_nums
        )

def check_series_properly_discretized(orig, tr, reconst, max_nums=10):
    orig_notna = orig.notna()
    tr_notna = tr.notna()
    reconst_notna = reconst.notna()
    assert (orig_notna == tr_notna).all()
    assert (orig_notna == reconst_notna).all()
    if pd.api.types.is_numeric_dtype(orig):
        if pd.api.types.is_categorical_dtype(tr):
            for val, interv, reconst in zip(orig[orig_notna], tr[tr_notna], reconst[reconst_notna]):
                assert val in interv
                assert reconst in interv
        else:
            assert orig.nunique() <= max_nums
            assert (orig[orig_notna] == tr[tr_notna]).all()
    else:
        assert (orig[orig_notna] == tr[tr_notna]).all()


@pytest.mark.parametrize('openml_id', [31, 1461, 40536])
def test_df_discretizer(openml_id):
    disc = pysynth.catdecat.DataFrameDiscretizer(max_num_cats=300)
    df = test_data.get_openml(openml_id)
    tr_df = disc.fit_transform(df)
    reconst_df = disc.inverse_transform(tr_df)
    check_df_properly_discretized(df, tr_df, reconst_df, max_nums=10)

