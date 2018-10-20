

import sklearn
import numpy
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emtrees


def test_basic_binary_classification():
    X, Y = datasets.make_classification(n_classes=2, n_samples=1000, random_state=1)
    trees = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    scores = model_selection.cross_val_score(trees, X, Y, scoring='accuracy')

    assert numpy.mean(scores) > 0.7, scores

def test_binary_classification_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    model = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    model.fit(X, Y)

    trees = emtrees.convert(model, method='loadable')
    predicted = trees.predict(X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.9 # testing on training data

def test_extratrees_classification_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    model = ExtraTreesClassifier(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    model.fit(X, Y)

    trees = emtrees.convert(model, method='loadable')
    predicted = trees.predict(X)
    accuracy = metrics.accuracy_score(Y, predicted)

    assert accuracy > 0.85 # testing on training data

def test_inline_compiled():
    X, Y = datasets.make_classification(n_classes=2, random_state=1)
    model = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    model.fit(X, Y)

    trees = emtrees.convert(model, method='inline')
    predicted = trees.predict(X)
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
    model = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=1)
    X = (X * 2**16).astype(int) # convert to integer
    model.fit(X, Y)

    trees = emtrees.convert(model)
    dot = trees.to_dot(name='ffoo')
    with open('tmp/trees.dot', 'w') as f:
        f.write(dot)
