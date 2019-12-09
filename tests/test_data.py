import pandas as pd
import sklearn.datasets

def get_openml(id):
    return sklearn.datasets.fetch_openml(
        data_id=id,
        target_column=None,
        as_frame=True,
    )['data']

def test_openml():
    colnames = [
        'fathers_occupation',
        'sons_occupation',
        'family_structure',
        'race',
        'counts_for_sons_first_occupation',
        'counts_for_sons_current_occupation'
    ]
    df = get_openml(541)
    assert df.columns.tolist() == colnames
    assert (df[colnames[:4]].dtypes == 'category').all()
    assert df[colnames[4:]].dtypes.apply(pd.api.types.is_numeric_dtype).all()
    assert len(df.index) == 1156

def check_synthdf_equal(df, synth, nrows=None):
    assert frozenset(df.columns) == frozenset(synth.columns)
    assert len(synth.index) == (len(df.index) if nrows is None else nrows)
    for col in df.columns:
        assert df[col].dtype == synth[col].dtype
        if pd.api.types.is_categorical_dtype(df[col].dtype):
            assert frozenset(synth[col].cat.categories).issubset(df[col].cat.categories)
        elif pd.api.types.is_object_dtype(df[col].dtype):
            assert frozenset(synth[col]).issubset(frozenset(df[col]))
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            assert df[col].min() <= synth[col].mean() <= df[col].max()
