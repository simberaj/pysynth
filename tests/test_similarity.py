import sys
import os
import itertools
import warnings

import scipy.stats
import numpy as np
import pandas as pd
import pytest
import sklearn.base
import sklearn.exceptions

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth.similarity
import test_data


def generate_artificial_df(n=200):
    return pd.DataFrame({
        'prob': np.random.rand(n),
        'iq': scipy.stats.norm(loc=100, scale=15).rvs(n),
        'cat' : np.random.choice(list('abc'), size=n),
    })


@pytest.fixture(scope='module')
def testing_df_close_pairs():
    np.random.seed(1711)
    pairs = [
        (generate_artificial_df(1000), generate_artificial_df(300)),
    ]
    for openml_id in [31, 1461, 40536]:
        df = test_data.get_openml(openml_id)
        half = len(df.index) // 2
        pairs.append((df.iloc[:half], df.iloc[half:]))
    return pairs


@pytest.mark.parametrize('distro_name, n_samples', itertools.product(
    ['norm', 'uniform'],
    [2, 30, 1000],
))
def test_summary_stats_artificial(distro_name, n_samples):
    distro = getattr(scipy.stats, distro_name)()
    data = pd.Series(distro.rvs(n_samples))
    sumstat = pysynth.similarity.summary_stats(data)
    for item in pysynth.similarity.SUMMARY_STATS:
        assert item in sumstat.index
    assert np.isclose(sumstat['mean'], data.mean())
    assert np.isclose(sumstat['skew'], data.skew()) or (
        np.isnan(sumstat['skew']) and np.isnan(data.skew())
    )

def test_summary_stat_diff_artificial():
    df1, df2 = generate_artificial_df(), generate_artificial_df()
    df_diff = pysynth.similarity.summary_stat_diff(df1, df2, 'diff')
    # check range for probs column
    assert (-1 < df_diff.loc['prob',:]).all()
    assert (df_diff.loc['prob',:] < 1).all()


@pytest.mark.parametrize('method', ['diff', 'ape'])
def test_summary_stat_diff_real(testing_df_close_pairs, method):
    for df1, df2 in testing_df_close_pairs:
        nodiff = pysynth.similarity.summary_stat_diff(df1, df2, method)
        # check all stats have their column
        assert frozenset(nodiff.columns) == frozenset([
            col + '_' + method for col in pysynth.similarity.SUMMARY_STATS
        ])
        # check all numeric columns have their row
        assert frozenset(nodiff.index) == frozenset([
            col for col in df2 if pd.api.types.is_numeric_dtype(df2[col])
        ])
        if df1 is df2:
            assert np.isclose(nodiff, 0).all()


@pytest.mark.parametrize('n_bins', [5, 10, 15])
def test_aligned_freqs_normal(n_bins):
    np.random.seed(1711)
    df1, df2 = generate_artificial_df(1000), generate_artificial_df(300)
    for col in df1.columns:
        f1, f2 = pysynth.similarity.aligned_freqs(df1[col], df2[col], n_bins)
        f1_cats = frozenset(f1.index)
        assert not f1.hasnans
        assert not f2.hasnans
        assert f1_cats == frozenset(f2.index)
        assert np.isclose(f1.sum(), 1)
        assert (f1 >= 0).all() and (f1 <= 1).all()
        assert np.isclose(f2.sum(), 1)
        assert (f2 >= 0).all() and (f2 <= 1).all()
        # since the generation process is the same, the diffs should be low
        assert abs(f1 - f2).max() < .1
        if pd.api.types.is_numeric_dtype(df1[col]):
            assert len(f1_cats) == n_bins
            # the frequencies in f1 should be appx equal due to quantile binning
            assert (abs(f1[f1 > 0] - 1 / len(f1_cats)) < .01).all()
        else:
            assert len(f1_cats) == df1[col].nunique()


def test_aligned_freqs_nobin():
    x, y = pd.Series(np.random.rand(300)), pd.Series(np.random.rand(100))
    f_x, f_y = pysynth.similarity.aligned_freqs(x, y, bins=None)
    assert f_x is None
    assert f_y is None


@pytest.mark.parametrize('bins, metrics', itertools.product(
    [5, 10, None], [None, ['rtae', 'mae']]
))
def test_frequency_mismatch_normal(testing_df_close_pairs, bins, metrics):
    for df1, df2 in testing_df_close_pairs:
        metric_df = pysynth.similarity.frequency_mismatch(df1, df2, bins, metrics)
        if metrics is None:
            metrics = list(freqdiff_metric_bounds.keys())
        assert frozenset(metric_df.columns) == frozenset(metrics)
        if bins is None:
            assert metric_df.index.tolist() == [
                col for col in df2 if not pd.api.types.is_numeric_dtype(df2[col])
            ]
        else:
            assert metric_df.index.tolist() == df2.columns.tolist()
        assert all(not metric_df[col].hasnans for col in metric_df.columns)

def random_freqs(n, zero_frac, index):
    if not (0 <= zero_frac < 1): raise ValueError
    freqs = np.random.rand(n)
    freqs[np.random.rand(n) < zero_frac] = 0
    return pd.Series(freqs / freqs.sum(), index=index)

freqdiff_metric_bounds = {
    'rtae': (0, 0, 2),
    'overlap_coef': (1, 0, 1),
    'rank_damerau': (0, 0, 1),
    'morisita_overlap': (1, 0, 1),
    'mae': (0, 0, 1),
    'rmse': (0, 0, 1),
    'jaccard_dist': (0, 0, 1),
    'simpson_diff': (0, -1, 1),
    'entropy_diff': (0, -np.inf, np.inf),
}

@pytest.mark.parametrize('n_cats, zero_frac, metrics', itertools.product(
    [2, 5, 15], [0, .2, .6], [None, ['rtae', 'mae']]
))
def test_freqdiff_metrics(n_cats, zero_frac, metrics):
    np.random.seed(1711)
    if n_cats == 2 and zero_frac == .6: return    # invalid case
    cats = ['c' + str(i) for i in range(n_cats)]
    probs1 = random_freqs(n_cats, zero_frac, cats)
    probs2 = random_freqs(n_cats, zero_frac, cats)
    metric_vals = pysynth.similarity.freqdiff_metrics(probs1, probs2, metrics)
    nodiff_vals = pysynth.similarity.freqdiff_metrics(probs1, probs1, metrics)
    if metrics is None:
        metrics = list(freqdiff_metric_bounds.keys())
    assert frozenset(metric_vals.index) == frozenset(metrics)
    assert pd.api.types.is_numeric_dtype(metric_vals)
    for metric, value in metric_vals.items():
        assert metric in freqdiff_metric_bounds
        nodiff, lo, hi = freqdiff_metric_bounds[metric]
        assert lo <= value <= hi
        assert metric in nodiff_vals.index
        assert np.isclose(nodiff_vals[metric], nodiff)


damlev_test_cases = [
    ([8, 3, 7], [8, 3, 7], 0),
    (['a', 'b', 'c'], ['a', 'b', 'c'], 0),
    (list(range(100)), list(range(100)), 0),
    (['c'], ['c'], 0),
    ([], [], 0),
    ([6, 4, 2], [4, 2], 1),
    ([6, 4, 2], [6, 4], 1),
    ([3, 8, 1], [], 3),
    ([3, 8, 1], [9, 1], 2),
    ([3, 8, 1], [8, 3, 1], 1),
    ([3, 8, 1], [1, 8, 3], 2),
    ([3, 8, 1], [3, 9, 1], 1),
]

def test_damerau_levenshtein():
    for seq1, seq2, dist in damlev_test_cases:
        assert pysynth.similarity.damerau_levenshtein(seq1, seq2) == dist
        assert pysynth.similarity.damerau_levenshtein(seq2, seq1) == dist


@pytest.mark.parametrize('method', ['pearson', 'kendall', 'spearman'])
def test_correlation_diff(testing_df_close_pairs, method):
    for df1, df2 in testing_df_close_pairs:
        corrdiff = pysynth.similarity.correlation_diff(df1, df2, method)
        ok_subset = []
        for col in df2.columns:
            if pd.api.types.is_numeric_dtype(df2[col]):
                assert col in corrdiff.index
                assert col in corrdiff.columns
                if not np.isnan(corrdiff[col]).all():
                    ok_subset.append(col)
        selfcorr_diffs = np.diag(corrdiff.loc[ok_subset,ok_subset].values)
        assert (selfcorr_diffs == 0).all()
        assert (np.isnan(corrdiff) | (corrdiff >= -1)).all(axis=None)
        assert (np.isnan(corrdiff) | (corrdiff <= 1)).all(axis=None)


def test_stat_tests(testing_df_close_pairs):
    for df1, df2 in testing_df_close_pairs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            stat_df = pysynth.similarity.stat_tests(df1, df2)
        assert stat_df.columns.tolist() == [
            't_stat', 't_pval', 'levene_stat', 'levene_pval'
        ]
        for col in df2.columns:
            if pd.api.types.is_numeric_dtype(df2[col]):
                assert col in stat_df.index
        for stat in stat_df.columns:
            if stat.endswith('pval'):
                assert (np.isnan(stat_df[stat]) | (stat_df[stat] >= 0)).all()
                assert (np.isnan(stat_df[stat]) | (stat_df[stat] <= 1)).all()


@pytest.mark.parametrize('openml_id', [31, 1461, 40536])
def test_predictor_matrix(openml_id):
    df = test_data.get_openml(openml_id)
    preds = pysynth.similarity._predictor_matrix(df)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.number)
    assert preds.ndim == 2
    assert preds.shape[0] == len(df.index)
    assert preds.shape[1] >= len(df.columns)


@pytest.mark.parametrize('metrics', [None, ['gini', 'f1', 'precision']])
def test_compute_accuracy_metrics(metrics):
    np.random.seed(1711)
    n = 200
    sources = np.random.rand(n)
    targets = sources < .5
    prob_variants = [
        (targets.astype(float), False),
        (sources, False),
        (1 - sources, False),
        (np.clip(sources + np.random.rand(n) * .2 - .1, 0, 1), False),
        (np.clip(sources + np.random.rand(n) * .4 - .2, 0, 1), False),
        (np.zeros(n), True),
        (np.ones(n), True),
    ]
    check_metrics = metrics
    if check_metrics is None:
        check_metrics = list(pysynth.similarity.DEFAULT_METRICS.keys())
    for probs, is_edge in prob_variants:
        if is_edge:
            with pytest.warns(None):
                metric_vals = pysynth.similarity._compute_accuracy_metrics(targets, probs, metrics)
        else:
            metric_vals = pysynth.similarity._compute_accuracy_metrics(targets, probs, metrics)
        check_metric_values(metric_vals, check_metrics)


def check_metric_values(metrics, names=None):
    if names is not None:
        assert metrics.index.tolist() == names
    assert not metrics.hasnans
    assert (np.round(metrics, 8) <= 1).all()
    assert (np.round(metrics, 8) >= -1).all()
    

@pytest.mark.parametrize('test_size', [.1, .25, .5])
def test_discrimination(testing_df_close_pairs, test_size):
    for df1, df2 in testing_df_close_pairs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
            metrics, clf = pysynth.similarity.discrimination(
                df1, df2, test_size=test_size, return_best=True
            )
        check_metric_values(metrics)
        if clf is None:
            assert metrics['gini'] <= 0
        else:
            assert metrics['gini'] > 0
            assert sklearn.base.is_classifier(clf)
