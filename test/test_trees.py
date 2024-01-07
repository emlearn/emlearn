
import os

import sklearn
import numpy
import numpy.testing
import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import emlearn
from emlearn.evaluate.trees import model_size_nodes
import pytest

here = os.path.dirname(__file__)

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))

# FIXME: add tests case for soft voting with leaf quantization. leaf_bits=2,4,6
# FIXME: add case with majority voting. leaf_bits=0?


# FIXME: add case with max_depth
CLASSIFICATION_MODELS = {
    'RFC': RandomForestClassifier(n_estimators=10, random_state=random),
    #'RFC-maxdepth': RandomForestClassifier(n_estimators=10, random_state=random, max_depth=2),
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

METHODS = ['loadable', 'inline']

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

    proba_original = estimator.predict_proba(X[:5])
    proba_c = cmodel.predict_proba(X[:5])
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=0.001)

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

@pytest.fixture(scope='module')
def huge_trees_model():
    store_classifier_path = os.path.join(here, 'out/test_trees_huge.model.pickle')
    X, Y = datasets.make_classification(n_classes=2, n_samples=1000, random_state=1)
    est = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=1)
    est.fit(X, Y)

    n_nodes = model_size_nodes(est)
    assert n_nodes >= 90*1000

    return X, Y, est

@pytest.mark.parametrize("method", ['loadable', 'inline'])
@pytest.mark.skipif(not bool(int(os.environ.get('EMLEARN_TESTS_SLOW', '0'))), reason='EMLEARN_TESTS_SLOW not enabled')
def test_trees_huge(method, huge_trees_model):
    """Should work just the same as a smaller model"""
    X, Y, estimator = huge_trees_model

    cmodel = emlearn.convert(estimator, method=method)
    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    numpy.testing.assert_equal(pred_c, pred_original)

def test_trees_to_dot():
    X, Y = datasets.make_classification(n_classes=2, n_samples=10, random_state=1)
    model = RandomForestClassifier(n_estimators=3, max_depth=5, random_state=1)
    model.fit(X, Y)

    trees = emlearn.convert(model)
    dot = trees.to_dot(name='ffoo')
    with open('tmp/trees.dot', 'w') as f:
        f.write(dot)
