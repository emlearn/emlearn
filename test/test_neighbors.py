
import emlearn
import eml_net

import pytest
import sklearn
import numpy
from numpy.testing import assert_equal, assert_almost_equal

import sys
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


UNSUPPORTED_PARAMS={
    # NOTE: update when supporting more distance metrics
    'p=1 (manhattan)': dict(p=1), # manhattan
    'metric=manhattan': dict(metric='manhattan'),
    'generic minowski': dict(p=2.23),
    'weighted distance': dict(weights='distance'),
    'custom distance': dict(weights=lambda x: x),
}

@pytest.mark.parametrize('name', UNSUPPORTED_PARAMS.keys())
def test_unsupported_classifier(name):
    params = UNSUPPORTED_PARAMS[name]
    estimator = KNeighborsClassifier(**params)

    estimator.fit([[1.0, 1.0]], [1])

    with pytest.raises(ValueError) as ex:
        emlearn.convert(estimator, method='loadable')

    assert 'Unsupported ' in str(ex.value)


def assert_equivalent(model, X_test, n_classes, method):
    cmodel = emlearn.convert(model, method=method)

    # TODO: support predict_proba, use that instead
    cpred = cmodel.predict(X_test)
    pred = model.predict(X_test)

    assert_equal(pred, cpred)



def make_classification_dataset(n_features=10, n_classes=10):

    rng = numpy.random.RandomState(0)
    X, y = make_classification(n_features=n_features, n_classes=n_classes,
                               n_redundant=0, n_informative=n_features,
                               random_state=rng, n_clusters_per_class=3, n_samples=50)
    X += 2 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)
    X = numpy.clip(1000*X, -32768, 32767).astype(int) # convert to int16 range

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    return X_train, X_test, y_train, y_test

SUPPORTED_PARAMS = {
    'default': dict(),
    'metric=euclidean': dict(metric='euclidean'),
    'NN1': dict(n_neighbors=1),
}

@pytest.mark.parametrize('params', SUPPORTED_PARAMS)
def test_classifier_predict(params):

    n_classes = 10
    p = SUPPORTED_PARAMS[params]
    model = KNeighborsClassifier(**p)
    print(params, p)

    X_train, X_test, y_train, y_test = make_classification_dataset(n_classes=n_classes)

    model.fit(X_train, y_train)

    # only test a subset
    X_test = X_test[:10]

    #assert_equivalent(model, X_test, params['classes'], method='pymodule')
    assert_equivalent(model, X_test, n_classes, method='loadable')


