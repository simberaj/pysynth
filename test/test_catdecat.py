import sys
import os
import itertools

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth.catdecat

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


@pytest.mark.parametrize('dist_cls', pysynth.catdecat.DISTRIBUTORS.values())
def test_distributors(dist_cls):
    distributor = dist_cls(seed=42)
    minval = 2
    maxval = 7
    for i in range(10):
        vals = pd.Series(np.random.rand(100) * (maxval - minval) + minval)
        distributor.fit(vals)
        reconst = distributor.generate(100)
        assert minval <= reconst.min() <= reconst.max() <= maxval


def test_mean_distributor():
    dist = pysynth.catdecat.MeanDistributor()
    for i in range(10):
        vals = np.random.rand(100)
        val_mean = vals.mean()
        dist.fit(vals)
        assert (dist.generate(20) == np.array([val_mean] * 20)).all()

@pytest.mark.parametrize('categ', [
    pysynth.catdecat.Categorizer(seed=42),
    pysynth.catdecat.Categorizer(binner='equalrange', distributor='mean', seed=42),
])
def test_categorizer_numeric(categ):
    minval = -3
    maxval = 10
    vals = pd.Series(np.random.rand(100) * 13 - 3)
    categ.fit(vals)
    check_properly_categorized(vals, categ)

@pytest.mark.parametrize('n_cats', [2, 20, 70])
def test_categorizer_category(n_cats):
    vals = pd.Series(np.random.choice([chr(48 + i) for i in range(n_cats)], 300))
    c = pysynth.catdecat.Categorizer(seed=42)
    if vals.nunique() > c.max_num_cats:
        with pytest.raises(ValueError):
            c.fit(vals)
    else:
        trans = c.fit_transform(vals)
        assert (trans == vals).all()


@pytest.mark.parametrize('n_vals', [2, 20, 70])
def test_categorizer_integer(n_vals):
    vals = pd.Series(np.random.randint(n_vals, size=300))
    c = pysynth.catdecat.Categorizer(seed=42)
    c.fit(vals)
    if n_vals < c.min_for_bin:
        assert (c.transform(vals) == vals).all()
    else:
        check_properly_categorized(vals, c)


def check_properly_categorized(vals, categ):
    cats = categ.fit_transform(vals)
    # check that all values lie in the delimited intervals
    for val, interv in zip(vals, cats):
        assert val in interv
    reconst = categ.inverse_transform(cats)
    # check that all reconstructed values lie in the intervals
    for interv, subser in reconst.groupby(cats):
        for item in subser:
            assert item in interv
    # for interv, mean in means.iteritems():
        # assert mean in interv
    