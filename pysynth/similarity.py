'''Measure the statistical similarity of synthesized data to the original.'''

from typing import Any, List, Optional, Tuple, Collection, Dict, Iterable, Union, Callable

import numpy as np
import scipy.stats
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree

from . import catdecat

SUMMARY_STATS: List[str] = [
    'mean',
    'std',
    'min',
    'q1',
    'median',
    'q3',
    'max',
    'skew',
    'kurt'
]


def summary_stats(series: pd.Series) -> pd.Series:
    '''Produce univariate summary statistics for a numerical series.

    Provides quartiles (q1, median and q3 respectively), mean, standard
    deviation (std), skewness (skew), kurtosis (kurt) and extremes (min, max).
    Note that for very short series, the higher moments (std, skew, kurt)
    might come out as NaN.

    :param series: A numerical series to compute summary statistics for.
    '''
    sumstat = series.describe().drop('count')
    # rename quartiles
    index = sumstat.index.tolist()
    index[index.index('25%'):index.index('75%')+1] = ['q1', 'median', 'q3']
    sumstat.index = index
    # add what pandas describe does not provide
    for key in SUMMARY_STATS:
        if key not in index:
            sumstat[key] = getattr(series, key)()
    return sumstat


DIFF_METHODS = {
    'diff': lambda orig, synth: synth - orig,
    'ape': lambda orig, synth: ((synth - orig) / orig).where(synth != orig, 0.),
}


def summary_stat_diff(orig: pd.DataFrame,
                      synth: pd.DataFrame,
                      method: str = 'diff',
                      ) -> pd.DataFrame:
    '''Compute differences of summary statistics for the synthesized dataset.

    For all numerical columns of the synthesized dataset, compute its summary
    statistics and compare them with the original using the given method.

    :param orig: The original dataset.
    :param synth: The synthesized dataset.
    :param method: The method to use for comparing the statistics:

        - `'diff'` for absolute difference,
        - `'ape'` for absolute percentage difference.
    :returns: A dataframe with a row for each column of the synthetic
        dataframe, with columns for different summary statistics
        containing their differences.
    '''
    method_fx = DIFF_METHODS[method]
    num_cols = [col for col in synth.columns
                if pd.api.types.is_numeric_dtype(synth[col])]
    diff_df = pd.DataFrame.from_records([
        method_fx(
            summary_stats(synth[col]),
            summary_stats(orig[col])
        ).rename(col)
        for col in num_cols
    ], index=num_cols)
    # add _diff to stat names in columns to be more descriptive
    return diff_df.rename(columns={
        stat: stat + '_' + method for stat in diff_df.columns
    })


def aligned_freqs(orig: pd.Series,
                  synth: pd.Series,
                  bins: Optional[int] = 10,
                  ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    '''Return relative frequencies of values in the original and synthesized series.

    The relative frequency series will be aligned so that all values from
    both columns are present in both outputs.

    :param orig: A column from the original dataframe.
    :param synth: The corresponding column from the synthesized dataframe.
    :param bins: Number of bins (quantiles) to which to discretize the
        columns if they are numeric. Numeric columns with less unique values
        than this number will not be discretized. The quantiles are measured
        on the original column. If this is None, Nones will be returned for
        both outputs if the columns are numeric.
    :returns: A tuple of relative frequency series (summing to 1) for the
        original and synthesized dataset respectively, or a tuple of two Nones,
        if the originals are numeric and number of bins is not set.
    '''
    if pd.api.types.is_numeric_dtype(synth):
        if bins is None:
            return None, None
        elif synth.nunique() > bins or orig.nunique() > bins:
            quantiles = (
                [min(orig.min(), synth.min()) - 1]
                + catdecat.QuantileBinner(bins).get(orig)
                + [max(orig.max(), synth.max()) + 1]
            )
            orig = pd.cut(orig, quantiles)
            synth = pd.cut(synth, quantiles)
    orig_counts = orig.value_counts(normalize=True)
    synth_counts = synth.value_counts(normalize=True)
    orig_counts, synth_counts = orig_counts.align(synth_counts)
    return orig_counts.fillna(0), synth_counts.fillna(0)


def frequency_mismatch(orig: pd.DataFrame,
                       synth: pd.DataFrame,
                       bins: Optional[int] = 10,
                       metrics: Optional[List[str]] = None,
                       ) -> pd.DataFrame:
    '''Return mismatch metrics for the dataframe columns' value frequencies.

    This only looks at univariate value frequencies, not considering whether
    the values occur in conjunction with "correct" values from other columns.

    Computes the following metrics:

    -   `rtae`: Relative Total Absolute Error (sum of absolute differences).
            Goes from 0 for perfect match to 2 for totally different values.
    -   `overlap_coef`: Overlap coefficient (magnitude of set-wise frequency
            intersection). Goes from 1 for perfect match to 0 for totally
            different values.
    -   `morisita_overlap`: Morisita's overlap index[#], a measure of frequency
            overlap. Goes from 0 for no overlap to 1 for identical proportions.
    -   `rank_damerau`: Normalized Damerau-Levenshtein distance[#] of
            frequency-ordered category sets for both datasets; essentially,
            a number of adjustments (additions, deletions, swaps) to arrive
            from one to the other. Goes from 0 for matching category
            ranks to 1 for total mismatch.
    -   `mae`: Mean Absolute Error (mean of absolute differences). The less,
            the better.
    -   `rmse`: Root Mean Square Error. The less, the better.
    -   `jaccard_dist`: Jaccard distance[#] (Intersection over Union) of the
            two frequency sets. `jaccard_dist = 1 - overlap_coef`
    -   `simpson_diff`: Difference between Simpson diversity indices[#] for the
            synthetic and original frequencies.
    -   `entropy_diff`: Difference between the Shannon entropy[#] for the
            synthetic and original frequencies, in nats.

    :param orig: The original dataframe.
    :param synth: The synthesized analog.
    :param bins: Number of bins to (quantiles) to which to discretize the
        columns if they are numeric, to be able to measure their frequencies
        as well. The quantiles are measured on the original column. If None,
        numeric columns will not be measured.
    :param metrics: Names of metrics to include. If None, all metrics are
        computed.
    :returns: A dataframe with a row for each column of the synthetic dataframe
        (except numeric columns if bins is None) with columns for different
        frequency mismatch statistics.

    [#] "Morisita's overlap index". Wikipedia.
        <https://en.wikipedia.org/wiki/Morisita%27s_overlap_index>
    [#] "Damerau-Levenshtein distance". Wikipedia.
        <https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance>
    [#] "Jaccard index". Wikipedia.
        <https://en.wikipedia.org/wiki/Jaccard_index>
    [#] "Simpson index". Wikipedia. In: Diversity index.
        <https://en.wikipedia.org/wiki/Diversity_index#Simpson_index>
    [#] "Shannon index". Wikipedia. In: Diversity index.
        <https://en.wikipedia.org/wiki/Diversity_index#Shannon_index>
    '''
    recs = []
    index = []
    for col in synth.columns:
        orig_freqs, synth_freqs = aligned_freqs(orig[col], synth[col], bins)
        if orig_freqs is not None and synth_freqs is not None:
            recs.append(freqdiff_metrics(orig_freqs, synth_freqs, metrics))
            index.append(col)
    return pd.DataFrame.from_records(recs, index=index)


def freqdiff_metrics(orig_freqs: pd.Series,
                     synth_freqs: pd.Series,
                     metrics: Optional[List[str]] = None,
                     ) -> pd.Series:
    '''Compute frequency mismatch metrics for two value frequency series.

    :param orig_freqs: Frequencies of values (or their intervals) in the
        original dataframe column.
    :param synth_freqs: Frequencies of values (or their intervals) in the
        matching synthesized column.
    :param metrics: Names of metrics to include. If None, all metrics are
        computed. For a list of metrics, see :func:`frequency_mismatch`.
    :returns: A Series with metric values, with their names in the index.
    '''
    diff = synth_freqs - orig_freqs
    simpson_orig = (orig_freqs ** 2).sum()
    simpson_synth = (synth_freqs ** 2).sum()
    overlap = orig_freqs.where(orig_freqs <= synth_freqs, synth_freqs)
    metric_series = pd.Series({
        'rtae': abs(diff).sum(),
        'overlap_coef': overlap.sum(),
        'rank_damerau': damerau_levenshtein(
            orig_freqs.sort_values().index.tolist(),
            synth_freqs.sort_values().index.tolist(),
        ) / len(orig_freqs.index),
        'morisita_overlap': (
            2 * (orig_freqs * synth_freqs).sum()
            / (simpson_orig + simpson_synth)
        ),
        'mae': abs(diff).mean(),
        'rmse': (diff ** 2).mean() ** .5,
        'jaccard_dist': 1 - overlap.sum(),
        'simpson_diff': simpson_synth - simpson_orig,
        'entropy_diff': (
            (synth_freqs[synth_freqs>0] * np.log(synth_freqs[synth_freqs>0])).sum()
            - (orig_freqs[orig_freqs>0] * np.log(orig_freqs[orig_freqs>0])).sum()
        )
    })
    if metrics is None:
        return metric_series
    else:
        return metric_series[metrics]


def damerau_levenshtein(seq1: Collection[Any], seq2: Collection[Any]) -> int:
    """Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    Based on code by Michael Homer, released under MIT License, retrieved
    from https://web.archive.org/web/20150909134357/http://mwh.geek.nz:80/2009/04/26/python-damerau-levenshtein-distance/
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            is_transposition = (
                x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]
            )
            if is_transposition:
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def correlation_diff(orig: pd.DataFrame,
                     synth: pd.DataFrame,
                     method: Union[str, Callable] = 'pearson',
                     ) -> pd.DataFrame:
    '''Return the difference of correlation matrices of the two dataframes.

    :param orig: The original dataframe.
    :param synth: The synthesized analog.
    :param method: A method for Pandas `corr()` to specify the manner of
        correlation (Pearson, Kendall, Spearman
        or arbitrary through a callable).
    :returns: A Dataframe with synthesized column names in the index and
        columns, with numerical differences of correlation coefficients in the
        dataframes as values. Might contain NaNs where the coefficient in either
        of the dataframes is NaN, e.g. when the given column only contains a
        single value.
    '''
    return (
        synth.corr(method=method)
        - orig[synth.columns.tolist()].corr(method=method)
    )


def stat_tests(orig: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    '''Test equality of mean and variance of synthesized columns to originals.

    Performs a two-sample independent t-test (`t_`) for mean equality
    and Levene's test for variance equality with median center (`levene_`),
    omitting NaNs.
    Omits non-numeric columns.

    :param orig: The original dataframe.
    :param synth: The synthesized analog.
    :returns: A dataframe with a row for each numeric column of the synthesized
        dataset, with a test statistic (`_stat`) and p-value (`_pval`) column
        for each of the tests performed.
    '''
    recs = []
    index = []
    for col in synth.columns:
        if pd.api.types.is_numeric_dtype(synth[col]):
            t, tp = scipy.stats.ttest_ind(
                orig[col], synth[col], nan_policy='omit'
            )
            if not isinstance(t, float):
                t, tp = np.nan, np.nan
            lev, levp = scipy.stats.levene(
                orig[col].dropna(), synth[col].dropna(), center='median'
            )
            recs.append((t, tp, lev, levp))
            index.append(col)
    return pd.DataFrame.from_records(
        recs, index=index,
        columns=['t_stat', 't_pval', 'levene_stat', 'levene_pval']
    )


DEFAULT_DISCRIMINATORS: List[sklearn.base.ClassifierMixin] = [
    sklearn.ensemble.GradientBoostingClassifier(n_estimators=10),
    sklearn.ensemble.RandomForestClassifier(n_estimators=10),
    # sklearn.linear_model.LogisticRegression(max_iter=250),
    # sklearn.linear_model.Perceptron(),
    # sklearn.linear_model.RidgeClassifier(),
    sklearn.naive_bayes.GaussianNB(),
    sklearn.neighbors.KNeighborsClassifier(),
    # sklearn.neighbors.RadiusNeighborsClassifier(),
    # sklearn.svm.LinearSVC(),
    # sklearn.svm.NuSVC(),
    # sklearn.svm.SVC(),
    sklearn.tree.DecisionTreeClassifier(),
]


def discrimination(orig: pd.DataFrame,
                   synth: pd.DataFrame,
                   classifiers: Iterable[sklearn.base.ClassifierMixin] = DEFAULT_DISCRIMINATORS,
                   metrics: Optional[List[str]] = None,
                   test_size: float = .25,
                   return_best: bool = False,
                   ) -> Union[
                       pd.Series,
                       Tuple[pd.Series, Optional[sklearn.base.ClassifierMixin]]
                   ]:
    '''Calculate how well the synthesized rows can be discriminated from originals.

    Fits each of the provided classifiers to predict whether the given row is
    synthesized or original, measures their accuracy on a test sample and
    gives a detailed evaluation of the best-performing one.

    :param orig: The original dataframe.
    :param synth: The synthesized analog.
    :param classifiers: Unfitted classifiers to try the discrimination. The
        one with the best ROC AUC on the test sample is selected.
    :param metrics: Names of discrimination accuracy metrics to compute.
        If None, all of these metrics are computed using their scikit-learn
        implementations:

        -   `auc`: ROC Area Under Curve (0.5 is no discrimination, 1 full
                discrimination).
        -   `gini`: ROC Gini coefficient (0 is no discrimination, 1 full
                discrimination: `gini = 2 * auc - 1`).
        -   `ap`: Average Precision (evaluates the precision-recall curve)[#].
        -   `matthews`: Matthews' four-square table correlation coefficient.
        -   `f1`: F1-score.
        -   `accuracy`: Ordinary accuracy (fraction of equally labeled rows).
        -   `precision`: Classification precision.
        -   `recall`: Classification recall.
        -   `cohen_kappa`: Cohen's kappa score of annotator agreement.
        -   `hamming`: Hamming loss.
        -   `jaccard`: Jaccard score.
    :param test_size: Fraction of the input to use for evaluating discrimination
        performance (and not for discriminator training). The train/test split
        is stratified on original/synthetic origins.
    :param return_best: Return the best performing fitted discriminator
        along with the metrics.
    :returns: A series of discrimination classification performance metrics
        with their aforementioned names in the index. If return_best is True,
        return a tuple with the metrics and the fitted discriminator.

    [#] "Average precision". Wikipedia. In: Information retrieval.
        <https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision>
    '''
    feats = _predictor_matrix(pd.concat([orig, synth]))
    print(feats)
    target = np.hstack((
        np.zeros(len(orig.index), dtype=bool),
        np.ones(len(synth.index), dtype=bool)
    ))
    best_est, best_probs, test_tgts = _find_best_classifier(
        feats, target, classifiers, test_size
    )
    metric_series = _compute_accuracy_metrics(test_tgts, best_probs, metrics)
    if return_best:
        return metric_series, best_est
    else:
        return metric_series


DEFAULT_METRICS: Dict[str, Tuple[Union[str, Callable], bool]] = {
    'auc': ('roc_auc_score', False),
    'gini': (
        (lambda trues, probs: 2 * sklearn.metrics.roc_auc_score(trues, probs) - 1),
        False
    ),
    'ap': ('average_precision_score', False),
    'matthews': ('matthews_corrcoef', True),
    'f1': ('f1_score', True),
    'accuracy': ('accuracy_score', True),
    'precision': ('precision_score', True),
    'recall': ('recall_score', True),
    'cohen_kappa': ('cohen_kappa_score', True),
    'hamming': ('hamming_loss', True),
    'jaccard': ('jaccard_score', True),
}


def _find_best_classifier(feats: np.ndarray,
                          target: np.ndarray,
                          classifiers: Iterable[sklearn.base.ClassifierMixin] = DEFAULT_DISCRIMINATORS,
                          test_size: float = .25,
                          ) -> Tuple[
                              sklearn.base.ClassifierMixin,
                              np.ndarray, np.ndarray
                          ]:
    train_feats, test_feats, train_tgts, test_tgts = \
        sklearn.model_selection.train_test_split(
            feats, target, test_size=test_size, stratify=target
        )
    best_auc = 0.5
    best_est = None
    best_probs = np.full_like(test_tgts, .5, dtype=np.double)
    for clf in classifiers:
        clf.fit(train_feats, train_tgts)
        probs = clf.predict_proba(test_feats)[:,1]
        auc = sklearn.metrics.roc_auc_score(test_tgts, probs)
        if auc > best_auc:
            best_est = clf
            best_auc = auc
            best_probs = probs
    return best_est, best_probs, test_tgts


def _compute_accuracy_metrics(targets: np.ndarray,
                              probs: np.ndarray,
                              metrics: Optional[List[str]] = None,
                              ) -> pd.Series:
    preds = probs >= .5
    metric_results = {}
    for name, conf in DEFAULT_METRICS.items():
        if metrics is None or name in metrics:
            fx, do_threshold = conf
            if isinstance(fx, str):
                fx = getattr(sklearn.metrics, fx)
            metric_results[name] = fx(
                targets,
                (preds if do_threshold else probs)
            )
    return pd.Series(metric_results)


def _predictor_matrix(dataframe: pd.DataFrame):
    dataframe = pd.get_dummies(
        dataframe,
        dummy_na=True,
    )
    fillers = {
        col: dataframe[col].median()
        for col in dataframe.columns
        if dataframe[col].hasnans
    }
    return dataframe.fillna(value=fillers).values
