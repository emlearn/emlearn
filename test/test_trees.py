
import sklearn
import numpy
import numpy.testing

from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emlearn
import pytest

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))

MODELS = {
    'RFC': RandomForestClassifier(n_estimators=10, random_state=random),
    'ETC': ExtraTreesClassifier(n_estimators=10, random_state=random),
}
DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=random),
}
METHODS = ['pymodule', 'loadable', 'inline']

@pytest.mark.parametrize("data", DATASETS.keys())
@pytest.mark.parametrize("model", MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_prediction_equals_sklearn(data, model, method):
    X, y = DATASETS[data]
    estimator = MODELS[model]

    X = (X * 2**16).astype(int) # currently only integers supported

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method)

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])

    numpy.testing.assert_equal(pred_c, pred_original)


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
    X = (X * 2**16).astype(int) # convert to integer
    model.fit(X, Y)

    trees = emlearn.convert(model)
    dot = trees.to_dot(name='ffoo')
    with open('tmp/trees.dot', 'w') as f:
        f.write(dot)
