
import os
import os.path
import subprocess

def get_include_dir():
    return os.path.join(os.path.dirname(__file__))


def build_classifier(cmodel, name, temp_dir, include_dir, func=None, compiler='cc', test_function=None):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if test_function is None:
        test_function = 'eml_test_read_csv'

    tree_name = name
    def_file_name = name+'.h'
    def_file = os.path.join(temp_dir, def_file_name)
    code_file = os.path.join(temp_dir, name+'.c')
    bin_path = os.path.join(temp_dir, name)

    # Trivial program that reads values on stdin, and returns classifications on stdout
    code = """
    #include "{def_file_name}"
    #include <eml_test.h>

    static void classify(const int32_t *values, int length, int row) {{
        const int32_t class = {func};
        printf("%d,%d\\n", row, class);
    }}
    int main() {{
        {test_function}(stdin, classify);
    }}
    """.format(**locals())

    with open(def_file, 'w') as f:
        f.write(cmodel)

    with open(code_file, 'w') as f:
        f.write(code)

    args = [
        compiler,
        '-std=c99',
        code_file, '-o', bin_path,
        '-I{}'.format(include_dir),
        '-I{}'.format(temp_dir),
    ]
    subprocess.check_call(args)

    return bin_path

def run_classifier(bin_path, data):
    lines = []
    for row in data:
        lines.append(",".join(str(v) for v in row))
    stdin = '\n'.join(lines)

    args = [ bin_path ]
    out = subprocess.check_output(args, input=stdin, encoding='utf8', universal_newlines=True)

    classes = []
    for line in out.split('\n'):
        if line:
            row,class_ = line.split(',')
            class_ = int(class_)
            classes.append(class_)

    assert len(classes) == len(data)

    return classes


class CompiledClassifier():
    def __init__(self, cmodel, name, call=None, include_dir=None, temp_dir='tmp/', test_function=None):
        if include_dir == None:
            include_dir = get_include_dir()
        self.bin_path = build_classifier(cmodel, name,
                include_dir=include_dir, temp_dir=temp_dir, func=call, test_function=test_function) 

    def predict(self, X):
        return run_classifier(self.bin_path, X)
