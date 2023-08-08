
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
    if n_classes == 2:
        pred = (pred[:,0] > 0.5).astype(int)
    else:
        pred = numpy.argmax(pred, axis=1)

    assert_equal(pred, cpred)


SUPPORTED_PARAMS = {
    'default': dict(),
    'metric=euclidean': dict(metric='euclidean'),
}

@pytest.mark.parametrize('params', SUPPORTED_PARAMS)
def test_classifier_predict(params):

    p = SUPPORTED_PARAMS[params]
    print(params, p)

    pass

