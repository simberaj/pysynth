import sys
import os
import tempfile
import shutil

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pysynth
import test_data

def test_synthesize():
    df = test_data.get_openml(469) # analcatdata_dmft
    synth = pysynth.synthesize(df)
    test_data.check_synthdf_equal(df, synth)

def test_main():
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, 'source.csv')
        out_path = os.path.join(tmp_dir, 'target.csv')
        test_data.get_openml(469).to_csv(in_path, sep=';', index=False)
        assert os.path.isfile(in_path)
        pysynth.main(in_path, out_path, '200')
        assert os.path.isfile(out_path)
        orig = pd.read_csv(in_path, sep=';')
        synth = pd.read_csv(out_path, sep=';')
        test_data.check_synthdf_equal(orig, synth, 200)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir)