
import emlearn
import eml_net

import pytest
import sklearn
import numpy
from numpy.testing import assert_equal, assert_almost_equal

import sys
import warnings
warnings.filterwarnings(action='ignore', category=sklearn.exceptions.ConvergenceWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler



def test_unsupported_activation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=10)
        model.fit([[1.0]], [True])
    with pytest.raises(Exception) as ex:
        model.activation = 'fake22'
        emlearn.convert(model)
    assert 'Unsupported activation' in str(ex.value)
    assert 'fake22' in str(ex.value)


def test_inference_simple():
    features = 2
    rng = numpy.random.RandomState(0)
    X, y = make_classification(n_features=features, n_classes=3,
                               n_redundant=0, n_informative=features,
                               random_state=rng, n_clusters_per_class=1)
    X += 2 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10, random_state=rng)
        model.fit(X, y)
        cmodel = emlearn.convert(model)
    
        X_test = X[:1]
        cpred = cmodel.predict_proba(X_test)
        pred = model.predict_proba(X_test)

    assert_almost_equal(cpred, pred, decimal=5)


PARAMS = [
    ( dict(hidden_layer_sizes=(4,), activation='relu'), {'classes': 3, 'features': 2}),
    ( dict(hidden_layer_sizes=(4,), activation='tanh'), {'classes': 2, 'features': 3}),
    ( dict(hidden_layer_sizes=(4,5,3)), {'classes': 5, 'features': 5}),
]

@pytest.mark.parametrize('modelparams,params', PARAMS)
def test_predict_equals_sklearn(modelparams,params):

    model = MLPClassifier(**modelparams, max_iter=20)

    for random in range(0, 3):
        # create dataset
        rng = numpy.random.RandomState(0)
        X, y = make_classification(n_features=params['features'], n_classes=params['classes'],
                                   n_redundant=0, n_informative=params['features'],
                                   random_state=rng, n_clusters_per_class=1, n_samples=50)
        X += 2 * rng.uniform(size=X.shape)
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

            cmodel = emlearn.convert(model)

            X_test = X_test[:3]
            cproba = cmodel.predict_proba(X_test)
            proba = model.predict_proba(X_test)
            cpred = cmodel.predict(X_test)
            pred = model.predict(X_test)

        assert_almost_equal(proba, cproba)
        assert_equal(pred, cpred)

# TODO: test matrix multiplication against numpy

