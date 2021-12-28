
import os
import sys
import subprocess
import platform
from distutils.ccompiler import new_compiler

def get_include_dir():
    return os.path.join(os.path.dirname(__file__))


def build_classifier(cmodel, name, temp_dir, include_dir, func=None, test_function=None):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if test_function is None:
        test_function = 'eml_test_read_csv'

    # create a new compiler object
    # force re-compilation even if object files exist (required)
    cc = new_compiler(force=1)

    tree_name = name
    def_file_name = name+'.h'
    def_file = os.path.join(temp_dir, def_file_name)
    code_file = os.path.join(temp_dir, name+'.c')
    output_filename = cc.executable_filename(name)
    bin_path = os.path.join(temp_dir, output_filename)
    include_dirs = [temp_dir, include_dir]
    if sys.platform.startswith('win'): # Windows
        libraries = None
        cc_args = None
    else : # MacOS and Linux should be the same
        libraries = ["m"] # math library / libm
        cc_args = ["-std=c99"]

    # Trivial program that reads values on stdin, and returns classifications on stdout
    code = """
    #include "{def_file_name}"
    #include <eml_test.h>

    static void classify(const float *values, int length, int row) {{
        printf("%d,%f\\n", row, (float){func});
    }}
    int main() {{
        {test_function}(stdin, classify);
    }}
    """.format(**locals())

    with open(def_file, 'w') as f:
        f.write(cmodel)

    with open(code_file, 'w') as f:
        f.write(code)
    objects = cc.compile(sources=[code_file],
        extra_preargs=cc_args, include_dirs=include_dirs)

    cc.link("executable", objects, output_filename=output_filename, 
        output_dir=temp_dir, libraries=libraries)  

    return bin_path

def run_classifier(bin_path, data, out_dtype='int'):
    lines = []
    for row in data:
        lines.append(",".join(str(v) for v in row))
    stdin = '\n'.join(lines)

    args = [ bin_path ]
    out = subprocess.check_output(args, input=stdin, encoding='utf8', universal_newlines=True)

    outputs = []
    for line in out.split('\n'):
        if line:
            row,out_ = line.split(',')
            if out_dtype == 'int':
                out_ = int(float(out_))
            elif out_dtype == 'float':
                out_ = float(out_)
            else:
                out_ = out_dtype(out_)
            outputs.append(out_)

    assert len(outputs) == len(data)

    return outputs

class CompiledClassifier():
    def __init__(self, cmodel, name, call=None, include_dir=None, temp_dir='tmp', test_function=None, out_dtype='int'):
        if include_dir == None:
            include_dir = get_include_dir()
        self.bin_path = build_classifier(cmodel, name,
                include_dir=include_dir, temp_dir=temp_dir, func=call, test_function=test_function) 
        self._out_dtype = out_dtype

    def predict(self, X):
        return run_classifier(self.bin_path, X, out_dtype=self._out_dtype)


def compile_executable(code_file,
                    out_dir,
                    name='main',
                    include_dirs=[]):

    cc = new_compiler(force=1)

    output_filename = cc.executable_filename(name)
    bin_path = os.path.join(out_dir, output_filename)
    #include_dirs = [out_dir, include_dir]

    if sys.platform.startswith('win'): # Windows
        libraries = None
        cc_args = None
    else: # MacOS and Linux should be the same
        libraries = ["m"] # math library / libm
        cc_args = ["-std=c99"]


    objects = cc.compile(
        sources=[code_file],
        extra_preargs=cc_args,
        include_dirs=include_dirs
    )

    cc.link("executable", objects,
        output_filename=output_filename, 
        output_dir=out_dir,
        libraries=libraries,
    )  

    return bin_path
