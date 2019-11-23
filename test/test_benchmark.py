
import io
import os.path
import subprocess

import pandas

import sys
from distutils.ccompiler import new_compiler

def test_bench_melspec():
    # create a new compiler object
    # force re-compilation even if object files exist (required)
    cc = new_compiler(force=1)
    output_filename = cc.executable_filename('bench')
    
    testdir = os.path.abspath(os.path.dirname(__file__))
    code = os.path.join(testdir, 'bench.c')
    prog = os.path.join(testdir, output_filename)
    print(prog)
    include_dirs=["emlearn"]

    if sys.platform.startswith('win'): # Windows
        # disable warnings: C5045-QSpectre, C4996-unsafe function/var
        cc_args = ["/Ox","/Oy-","/Wall","/WX","/wd5045","wd4996"]
        libraries = None
    else : # MacOS and Linux should be the same
        cc_args = ["-O3","-fno-omit-frame-pointer","-Wall","-Werror"]
        libraries = ["m"] # math library / libm

    objects = cc.compile(sources=[code], include_dirs=include_dirs,
     debug=1, extra_preargs=cc_args)
    cc.link("executable", objects, output_filename=output_filename, 
        debug=1, libraries=libraries, output_dir=testdir)
    out = subprocess.check_output([prog]).decode('utf-8')

    df = pandas.read_csv(io.StringIO(out), sep=';')
    melspec_time = df[df.task == 'melspec'].iloc[0].avg_time_us
    assert 10.0 < melspec_time < 10*1000
