
import subprocess
import os
import os.path

import sklearn
import numpy
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emtrees



def build_classifier(estimator, name='test_trees', temp_dir='tmp/', func=None):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    tree_name = name
    if func is None:
      func = 'emtrees_predict(&{}, values, length)'.format(tree_name)
    def_file = os.path.join(temp_dir, name+'.def.h')
    code_file = os.path.join(temp_dir, name+'.c')
    bin_path = os.path.join(temp_dir, name)

    # Trivial program that reads values on stdin, and returns classifications on stdout
    code = """
    #include "emtrees_test.h"
    #include "{def_file}"

    static void classify(const EmtreesValue *values, int length, int row) {{
        const int32_t class = {func};
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
    X, Y = datasets.make_classification(n_classes=2, n_samples=1000, random_state=1)
    trees = emtrees.RandomForest(n_estimators=10, max_depth=10, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    scores = model_selection.cross_val_score(trees, X, Y, scoring='accuracy')

    assert numpy.mean(scores) > 0.7, scores

def test_binary_classification_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    trees = emtrees.RandomForest(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    trees.fit(X, Y)

    p = build_classifier(trees)
    predicted = run_classifier(p, X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.9 # testing on training data

def test_extratrees_classification_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    trees = emtrees.ExtraTrees(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    trees.fit(X, Y)

    p = build_classifier(trees)
    predicted = run_classifier(p, X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.85 # testing on training data

def test_inline_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    trees = emtrees.RandomForest(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    trees.fit(X, Y)

    p = build_classifier(trees, 'myinline', func='myinline_predict(values, length)')
    predicted = run_classifier(p, X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.9 # testing on training data


def test_deduplicate_single_tree():
    nodes = [
        [ -1, 1, -1, -1 ],
        [ -1, 0, -1, -1 ],
        [ 2, 666, 0, 1 ],
        [ -1, 1, -1, -1 ], # dup leaf. idx=3
        [ 4, 333, 1, 3 ], # dup ref
        [ 5, 444, 2, 1],
        [ 6, 555, 4, 5],
    ]
    roots = [ 6 ]

    de_nodes, de_roots = emtrees.randomforest.remove_duplicate_leaves((nodes, roots))

    duplicates = 1
    assert len(de_roots) == len(roots)
    assert len(de_nodes) == len(nodes) - duplicates
    assert de_roots[0] == roots[0] - duplicates

def test_trees_to_dot():
    X, Y = datasets.make_classification(n_classes=2, n_samples=10, random_state=1)
    trees = emtrees.RandomForest(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    trees.fit(X, Y)

    dot = trees.to_dot(name='ffoo')
    with open('tmp/trees.dot', 'w') as f:
        f.write(dot)
