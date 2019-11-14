
import os.path
import subprocess
import json

import numpy

examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples'))


def run_window_function(options):
    path = os.path.join(examples_dir, 'window-function.py')
    args = ['python', path]

    for key, value in options.items():
        args.append('--{}={}'.format(key, value))

    stdout = subprocess.check_output(' '.join(args), shell=True)
    return stdout.decode('utf-8')


def run_extract(include, name, length, workdir, compiler='gcc'):
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

    prog_path = os.path.join(workdir, 'test_' + name)
    code_path = prog_path + '.c'

    params = dict(include=include, name=name, length=length)
    prog = template_prog.format(**params)
    with open(code_path, 'w') as f:
        f.write(prog)

    # compile
    args = [compiler, '-std=c99', code_path, '-o', prog_path]
    subprocess.check_call(' '.join(args), shell=True)

    stdout = subprocess.check_output([prog_path])
    arr = json.loads(stdout)
    return arr


def window_function_test(file_path, args, compiler='gcc'):
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
    arr = run_extract(os.path.abspath(file_path), args['name'], args['length'], out_dir, compiler=args['compiler'])
    return arr


def test_window_function_hann(compiler='gcc'):

    file_path = 'tests/out/window_func.h'
    args = dict(
        window='hann',
        length=512,
        name='some_name_for_array',
        out=file_path,
    )

    arr = window_function_test(file_path, args, compiler=compiler)
    # Hann has sum of coefficients == half of N
    numpy.testing.assert_allclose(numpy.sum(arr), args['length']/2)
