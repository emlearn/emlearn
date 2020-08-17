
import os.path
import subprocess
import json

import numpy
from distutils.ccompiler import new_compiler


def run_window_function(options):
    module = 'emlearn.tools.window_function'
    args = ['python', '-m', module]

    for key, value in options.items():
        args.append('--{}={}'.format(key, value))

    stdout = subprocess.check_output(' '.join(args), shell=True)
    return stdout.decode('utf-8')


def run_extract(include, name, length, workdir):
    template_prog = """

    #include <stdio.h>
    #include "{include}"

    void print_json(const float *arr, int length) {{
        // write as JSON array
        printf("[");
        for (int i=0; i<length; i++) {{
            printf("%f%s", arr[i], (i != length-1) ? ", " : "");
        }}
        printf("]");
    }}

    int main() {{

        const float *arr = {name};
        const int length = {length};

        print_json(arr, length);
    }}
    """
    # create a new compiler object
    # force re-compilation even if object files exist (required)
    cc = new_compiler(force=1)
    output_filename = cc.executable_filename('test_' + name)
    prog_path = os.path.join(workdir, output_filename)
    code_path = prog_path + '.c'

    params = dict(include=include, name=name, length=length)
    prog = template_prog.format(**params)
    with open(code_path, 'w') as f:
        f.write(prog)

    # compile
    objects = cc.compile([code_path])
    cc.link("executable", objects, output_filename=output_filename, 
        output_dir=workdir)  

    stdout = subprocess.check_output([prog_path])

    arr = json.loads(stdout)
    return arr


def window_function_test(file_path, args):
    out_dir = os.path.dirname(file_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(file_path):
        os.remove(file_path)

    stdout = run_window_function(args)

    assert file_path in stdout, 'output filename should be mentioned on stdout'
    assert os.path.exists(file_path), 'output file should have been written'

    with open(file_path, 'r') as f:
        contents = f.read()
        assert args['name'] in contents
        assert str(args['length']) in contents

    # check it compiles
    arr = run_extract(os.path.abspath(file_path), args['name'], args['length'], out_dir)
    return arr


def test_window_function_hann():

    file_path = os.path.join('tests','out','window_func.h')
    args = dict(
        window='hann',
        length=512,
        name='some_name_for_array',
        out=file_path,
    )

    arr = window_function_test(file_path, args)
    # Hann has sum of coefficients == half of N
    numpy.testing.assert_allclose(numpy.sum(arr), args['length']/2)
