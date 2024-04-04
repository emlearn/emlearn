
"""
Utilities
=========================
"""

import os
import sys
import subprocess
import platform
from distutils.ccompiler import new_compiler

import numpy

def check_array(arr):
    # import dynamically to not need this at package build time
    from sklearn.utils import check_array as check
    return check(arr)

def get_include_dir() -> str:
    """
    Get the include directory with C headers for emlearn
    """
    return os.path.join(os.path.dirname(__file__))


def build_classifier(cmodel, name, temp_dir, include_dir, func=None, test_function=None, n_classes=None):
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

    if n_classes is not None:
        code = """
        #include "{def_file_name}"
        #include <eml_test.h>

        #define N_CLASSES {n_classes}
        float outputs[N_CLASSES];

        static void classify_proba(const float *values, int length, int row) {{
            const EmlError err = {func};
            for (int class_no=0; class_no<N_CLASSES; class_no++) {{
                const float prob = outputs[class_no];
                printf("%d,%d,%f\\n", row, class_no, prob);
            }}
        }}
        int main() {{
            {test_function}(stdin, classify_proba);
        }}
        """.format(**locals())
    else:
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

def run_classifier(bin_path, data, out_dtype='int', float_precision=8):

    # Serialize input data as CSV
    def serialize_value(v):
        return '{:.{prec}f}'.format(v, prec=float_precision)

    lines = []
    for row in data:
        lines.append(",".join(serialize_value(v) for v in row))
    stdin = '\n'.join(lines)

    assert len(lines) == len(data), (len(lines), data.shape)

    # Run as subprocess
    args = [ bin_path ]
    out = subprocess.check_output(args, input=stdin, encoding='utf8', universal_newlines=True)

    # Parse output
    outputs = []
    lines = out.split('\n')
    for line in lines:
        if line:
            tokens = line.split(',')
            row = tokens[0]
            if len(tokens) == 2:
                out_ = tokens[1]
            else:
                out_ = tokens

            if out_dtype == 'int':
                out_ = int(float(out_))
            elif out_dtype == 'float':
                out_ = float(out_)
            else:
                out_ = out_dtype(out_)
            outputs.append(out_)

    return outputs

class CompiledClassifier():
    def __init__(self, cmodel, name, call=None, include_dir=None, temp_dir='tmp',
            test_function=None,
            out_dtype='int',
            proba_call=None,
            n_classes=None):

        if include_dir == None:
            include_dir = get_include_dir()
        self.bin_path = build_classifier(cmodel, name,
                include_dir=include_dir, temp_dir=temp_dir, func=call, test_function=test_function)

        self.proba_bin_path = None
        if proba_call is not None:
            self.proba_bin_path = build_classifier(cmodel, name+'_proba',
                    include_dir=include_dir, temp_dir=temp_dir,
                    func=proba_call, n_classes=n_classes,
                    test_function=test_function)

        self._out_dtype = out_dtype
        self.n_classes = n_classes

    def predict(self, X):
        X = check_array(X)

        out = run_classifier(self.bin_path, X, out_dtype=self._out_dtype)
        assert len(out) == len(X), out
        return out

    def predict_proba(self, X):
        X = check_array(X)

        if self.proba_bin_path is None:
            raise ValueError('predict_proba() not supported')

        def convert_out(raw):
            row, cls, prob = raw 
            return int(row), int(cls), float(prob)

        result = run_classifier(self.proba_bin_path, X, out_dtype=convert_out)
        out = numpy.empty(shape=(len(X), self.n_classes))
        for i, (row, cls, prob) in enumerate(result):
            out[row][cls] = prob

        assert len(out) == len(X), out
        return out

    def regress(self, X):
        X = check_array(X)

        return self.predict(X)



def compile_executable(code_file : str,
                    out_dir : str,
                    name : str ='main',
                    include_dirs=[]):
    """
    Compile C code on the host.

    Useful to integrate small C executables in a Python-based script or notebook.
    Uses distutil.ccompiler, same as what is used to build Python modules.
    Should work portably on all platforms.

    :param code_file: Path to file with C code to compile
    :param out_dir: Path to directory where output executable will be located
    :param name: Base name of the executable
    :param include_dirs: Include directories for C headers   

    :return: Path to executable
    """

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
