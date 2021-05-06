

import subprocess
import os
import os.path

import sklearn
import numpy
import numpy.testing
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emlearn

import pytest

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))

MODELS = {
    'GMM-full': GaussianMixture(n_components=3, covariance_type='full'),
    #'GMM-tied': GaussianMixture(n_components=3, covariance_type='tied'),
    #'GMM-diag': GaussianMixture(n_components=3, covariance_type='diag'),
    #'GMM-spherical': GaussianMixture(n_components=3, covariance_type='spherical'),
    'EllipticEnvelope': EllipticEnvelope(),
}
DATASETS = {
    #'binary': datasets.make_classification(n_classes=2, n_features=7, n_samples=100, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_features=7, n_informative=5, n_samples=100//2, random_state=random),
}
METHODS = [
    'inline',
    #'pymodule',
    #'loadable',
]


@pytest.mark.parametrize("data", DATASETS.keys())
@pytest.mark.parametrize("model", MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_gaussian_mixture_equals_sklearn(data, model, method):
    X, y = DATASETS[data]
    estimator = MODELS[model]

    X = preprocessing.StandardScaler().fit_transform(X)
    X = decomposition.PCA(3).fit_transform(X)
    estimator.fit(X, y)

    cmodel = emlearn.convert(estimator, method=method)

    if 'EllipticEnvelope' in model:
        dist = estimator.mahalanobis(X)    
        cdist = cmodel.mahalanobis(X)
        numpy.testing.assert_allclose(cdist, dist, rtol=1e-5)

    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    estimator.fit(X, y)
  
    numpy.testing.assert_equal(pred_c, pred_original)

