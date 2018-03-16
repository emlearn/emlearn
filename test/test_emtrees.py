
import sklearn
import numpy
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import emtrees
import subprocess

def build_classifier(estimator):
    def_file = 'tmp/test_trees.def.h'
    tree_name = 'test_trees'
    with open(def_file, 'w') as f:
        f.write(estimator.output_c(tree_name))
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

    code_file = 'tmp/test_trees.c'
    with open(code_file, 'w') as f:
        f.write(code)

    bin_path = 'tmp/test_trees'

    args = [ 'cc', code_file, '-o', bin_path, '-I./test', '-I.' ]
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

