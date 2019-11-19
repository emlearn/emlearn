
import io
import os.path
import subprocess

import pandas

import sys
from distutils.ccompiler import new_compiler

def test_bench_melspec():
    # create a new compiler object
    cc = new_compiler()
    output_filename = cc.executable_filename('bench')
    
    testdir = os.path.dirname(__file__)
    code = os.path.join(testdir, 'bench.c')
    prog = os.path.join(testdir, output_filename)
    include_dirs=["emlearn"]

    if sys.platform.startswith('win'): # Windows
        cc_args = ["/Ox","/Zi","/Oy-","/Wall","/WX"]
    else : # MacOS and Linux should be the same
        cc_args = ["-O3","-g","-fno-omit-frame-pointer","-Wall","-Werror"]

    objects = cc.compile([code], output_dir=testdir, include_dirs=include_dirs,
        debug=1)
    cc.link("executable", objects, output_filename=output_filename, 
        debug=1, output_dir=testdir)
    out = subprocess.check_output([prog]).decode('utf-8')

    df = pandas.read_csv(io.StringIO(out), sep=';')
    melspec_time = df[df.task == 'melspec'].iloc[0].avg_time_us
    assert 10.0 < melspec_time < 10*1000
