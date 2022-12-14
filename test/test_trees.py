
import sklearn
import numpy
import numpy.testing

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import emlearn
import pytest

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))

CLASSIFICATION_MODELS = {
    'RFC': RandomForestClassifier(n_estimators=10, random_state=random),
    'ETC': ExtraTreesClassifier(n_estimators=10, random_state=random),
    'DTC': DecisionTreeClassifier(random_state=random),
}

REGRESSION_MODELS = {
    'RFR': RandomForestRegressor(n_estimators=10, random_state=random),
    'ERR': ExtraTreesRegressor(n_estimators=10, random_state=random),
    'DTR': DecisionTreeRegressor(random_state=random),
}

CLASSIFICATION_DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=random),
}

REGRESSION_DATASETS = {
    '1out': datasets.make_regression(n_targets=1, n_samples=100, random_state=random),
}

METHODS = ['pymodule', 'loadable', 'inline']

@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", CLASSIFICATION_MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_trees_sklearn_classifier_predict(data, model, method):
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = CLASSIFICATION_MODELS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method)

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])

    numpy.testing.assert_equal(pred_c, pred_original)

@pytest.mark.parametrize("data", REGRESSION_DATASETS.keys())
@pytest.mark.parametrize("model", REGRESSION_MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_trees_sklearn_regressor_predict(data, model, method):
    X, y = REGRESSION_DATASETS[data]
    estimator = REGRESSION_MODELS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method)

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])

    numpy.testing.assert_allclose(pred_c, pred_original, rtol=1e-3, atol=2)


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

    de_nodes, de_roots = emlearn.trees.remove_duplicate_leaves((nodes, roots))

    duplicates = 1
    assert len(de_roots) == len(roots)
    assert len(de_nodes) == len(nodes) - duplicates
    assert de_roots[0] == roots[0] - duplicates

def test_trees_to_dot():
    X, Y = datasets.make_classification(n_classes=2, n_samples=10, random_state=1)
    model = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=1)
    model.fit(X, Y)

    trees = emlearn.convert(model)
    dot = trees.to_dot(name='ffoo')
    with open('tmp/trees.dot', 'w') as f:
        f.write(dot)
