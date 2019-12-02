import sys
import os
import itertools

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth.catdecat

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


def test_categorizer():
    minval = -3
    maxval = 10
    vals = pd.Series(np.random.rand(100) * (maxval - minval) + minval)
    c = pysynth.catdecat.Categorizer(seed=42)
    cated = c.fit_transform(vals)
    # check that all values lie in the delimited intervals
    for val, interv in zip(vals, cated):
        assert val in interv
    reconst = c.inverse_transform(cated)
    means = reconst.groupby(cated).mean()
    # check that means of reconstructed values lie in the intervals
    for interv, mean in means.iteritems():
        assert mean in interv
