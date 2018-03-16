
import subprocess
import os
import os.path

import sklearn
import numpy
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import emtrees



def build_classifier(estimator, name='test_trees', temp_dir='tmp/'):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    tree_name = name
    def_file = os.path.join(temp_dir, name+'.def.h')
    code_file = os.path.join(temp_dir, name+'.c')
    bin_path = os.path.join(temp_dir, name)

    # Trivial program that reads values on stdin, and returns classifications on stdout
    code = """
    #include "emtrees_test.h"
    #include "{def_file}"

    static void classify(const EmtreesValue *values, int length, int row) {{
        const int32_t class = emtrees_predict(&{tree_name}, values, length);
        printf("%d,%d\\n", row, class);
    }}
    int main() {{
        emtrees_test_read_csv(stdin, classify);
    }}
    """.format(**locals())

    with open(def_file, 'w') as f:
        f.write(estimator.output_c(tree_name))

    with open(code_file, 'w') as f:
        f.write(code)

    args = [ 'cc', '-std=c99', code_file, '-o', bin_path, '-I./test', '-I.' ]
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

def test_basic_binary_classification():
    X, Y = datasets.make_classification(n_classes=2)
    #trees = RandomForestClassifier(n_estimators=3, max_depth=5)
    trees = emtrees.RandomForest(n_trees=3, max_depth=5)
    X = (X * 2**16).astype(int) # convert to integer
    scores = model_selection.cross_val_score(trees, X, Y, scoring='accuracy')

    assert numpy.mean(scores) > 0.7, scores

def test_binary_classification_compiled():
    X, Y = datasets.make_classification(n_classes=2)
    trees = emtrees.RandomForest(n_trees=3, max_depth=5)
    X = (X * 2**16).astype(int) # convert to integer
    trees.fit(X, Y)

    p = build_classifier(trees)
    predicted = run_classifier(p, X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.9 # testing on training data

