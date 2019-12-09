from typing import List, Optional

import pandas as pd

from . import ipf

SYNTHESIZERS = {
    'ipf': ipf.IPFSynthesizer,
}

DEFAULT_METHOD = 'ipf'


def synthesize(dataframe: pd.DataFrame,
               n_rows: Optional[int] = None,
               method: str = DEFAULT_METHOD,
               ignore_cols: List[str] = [],
               **kwargs) -> pd.DataFrame:
    '''Synthesize an analog to a given dataframe.

    Optional keyword arguments are passed to the selected synthesizer.

    :param dataframe: Data to be synthesized.
    :param n_rows: Number of output rows. If omitted, the same
        length as the input dataframe will be used.
    :param method: Method to use for synthesis. So far, only the `ipf` method
        using :class:`ipf.IPFSynthesizer` is available.
    :param ignore_cols: Columns not to be synthesized in the output (such as
        personal identifiers).
    '''
    synther = SYNTHESIZERS[method](ignore_cols=ignore_cols, **kwargs)
    synther.fit(dataframe)
    return synther.sample(n_rows)


def main(in_file: str,
         out_file: str,
         n_rows: str = None,
         method: str = DEFAULT_METHOD
         ) -> None:
    '''Synthesize an analog to a given CSV file.

    :param in_file: A CSV file with data to serve as basis for synthesis.
    :param out_file: A path to output the synthesized CSV. Will be
        semicolon-delimited.
    :param n_rows: Number of rows for the output file. If omitted, the same
        length as the input file will be used.
    :param method: Synthesis method to be used (see :func:`synthesize`).
    '''
    if n_rows is not None:
        n_rows = int(n_rows)
    orig_df = pd.read_csv(in_file, sep=None, engine='python')
    synth_df = synthesize(orig_df, n_rows=n_rows, method=method)
    synth_df.to_csv(out_file, sep=';', index=False)
