

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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn import metrics
from sklearn.utils.estimator_checks import check_estimator 

import emlearn

import pytest

here = os.path.dirname(__file__)
random = numpy.random.randint(0, 1000)
random = 1
print('random_state={}'.format(random))

MODELS = {
    'GMM-full': GaussianMixture(n_components=3, covariance_type='full', random_state=random),
    'GMM-tied': GaussianMixture(n_components=3, covariance_type='tied', random_state=random),
    'GMM-diag': GaussianMixture(n_components=3, covariance_type='diag', random_state=random),
    'GMM-spherical': GaussianMixture(n_components=3, covariance_type='spherical', random_state=random),
    'B-GMM-full': BayesianGaussianMixture(n_components=2, covariance_type='full', random_state=random),
    'B-GMM-tied': BayesianGaussianMixture(n_components=5, covariance_type='tied', random_state=random),
    'B-GMM-diag': BayesianGaussianMixture(n_components=10, covariance_type='diag', random_state=random),
    'B-GMM-spherical': BayesianGaussianMixture(n_components=10, covariance_type='spherical', random_state=random),
    'EllipticEnvelope': EllipticEnvelope(),
}
DATASETS = {
    #'binary': datasets.make_classification(n_classes=2, n_features=7, n_samples=100, random_state=random),
    #'5way': datasets.make_classification(n_classes=2, n_features=4, n_informative=2, n_samples=6, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_features=7, n_informative=5, n_samples=50, random_state=random),
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
    X = decomposition.PCA(3, random_state=random).fit_transform(X)
    estimator.fit(X, y)

    cmodel = emlearn.convert(estimator, method=method)

    dist = estimator.score_samples(X)    
    cdist = cmodel.score_samples(X)
    numpy.testing.assert_allclose(cdist, dist, rtol=1e-5)

    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    estimator.fit(X, y)
  
    numpy.testing.assert_equal(pred_c, pred_original)

    # check that code can be generated
    out_dir = os.path.join(here, 'out', 'gaussian_mixture', f'{data}_{model}_{method}.c')
    save_path = os.path.join(out_dir, f'{data}_{model}_{method}.c')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cmodel.save(file=save_path, name='my_test_model')

