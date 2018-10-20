
import subprocess
import os
import os.path

import sklearn
import numpy
import numpy.testing
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emlearn
import eml_bayes

import pytest

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))

MODELS = {
    'GaussianNB': GaussianNB(),
}
DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=random),
}
METHODS = ['pymodule', 'loadable']

@pytest.mark.parametrize("data", DATASETS.keys())
@pytest.mark.parametrize("model", MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_bayes_equals_sklearn(data, model, method):
    X, y = DATASETS[data]
    estimator = MODELS[model]

    X = preprocessing.StandardScaler().fit_transform(X)
    estimator.fit(X, y)

    cmodel = emlearn.convert(estimator, method=method)
    
    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])
    numpy.testing.assert_equal(pred_c, pred_original)

