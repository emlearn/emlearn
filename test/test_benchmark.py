
import io
import os.path
import subprocess

import pandas

def compile_program(input, out, compiler='gcc'):
    args = [
        compiler,
        input,
        '-o', out,
        '-std=c99',
        '-O3',
        '-g',
        '-fno-omit-frame-pointer',
        '-lm',
        '-Wall',
        '-Werror',
        '-I./emlearn',
    ]
    subprocess.check_call(' '.join(args), shell=True)

def test_bench_melspec(compiler='gcc'):
    testdir = os.path.dirname(__file__)
    code = os.path.join(testdir, 'bench.c')
    prog = os.path.join(testdir, 'bench')
    compile_program(code, prog, compiler=compiler)
    out = subprocess.check_output([prog]).decode('utf-8')

    df = pandas.read_csv(io.StringIO(out), sep=';')
    melspec_time = df[df.task == 'melspec'].iloc[0].avg_time_us
    assert 10.0 < melspec_time < 10*1000
