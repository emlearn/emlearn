
import os.path
import os

import numpy

from . import common

# Ref 
"""
References

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_gaussian_mixture.py#L380

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_base.py

_estimate_log_gaussian_prob

log probability is used
implementation depends on covariance type. Can be known at fit time

Looks like log_det can be known at fit time
one constant per component

Stores
means_ means of each component
weights_ weights of each component

precisions_cholesky_
which is a Cholesky decomposition of the precision, the inverse of the covariance matrix

for full. components, features, features
for spherical. components
for diag. components, features
for tied, features, features 

BaysianGaussianMixture
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_bayesian_mixture.py

Seem to reuse _estimate_log_gaussian_prob from GMM
Has an additional term log_lambda
does not seem to depend on X?


https://www.vlfeat.org/api/gmm.html
implements only diagonal covariance matrix

https://github.com/vlfeat/vlfeat/blob/master/vl/gmm.c#L712

"""

from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
from sklearn.utils.extmath import row_norms
np = numpy


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))

        print('mm', n_samples, n_features, n_components)

        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):

            print("ss", X.shape, mu.shape, prec_chol.shape)

            if False:

                y = np.matmul(X, prec_chol) - np.matmul(mu, prec_chol)
                print("yy", y.shape)
                p = np.sum(np.square(y), axis=1) # sum over features
                print("p", p.shape, p)
                log_prob[:, k] = p          

            else:
                # sample iteration can be moved to outside
                for i, x in enumerate(X):
                    print('x', i, x.shape)

                    if True:
                        y = np.dot(x, prec_chol) - np.dot(mu, prec_chol)
                        print("yy", y.shape)
                        p = np.sum(np.square(y), axis=0) # sum over features
                        print("p", p.shape, p)

                    #for f_idx in range(x.shape):
                        
                        log_prob[i, k] = p

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))

    print('kk', log_prob.shape, log_det.shape)
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

class Wrapper:
    def __init__(self, estimator, classifier, dtype='float'):
        self.dtype = dtype


        n_components, n_features = estimator.means_.shape
        print("est shape", n_components, n_features)
        covariance_type = estimator.covariance_type
        precisions_chol = estimator.precisions_cholesky_

        from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky

        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features)

        #print("log_det", log_det.shape)
        #print("means", estimator.means_.shape)
        #print("prec", precisions_chol.shape)

        self._log_det = log_det
        self._means = estimator.means_.copy()
        self._covariance_type = covariance_type
        self._precisions_col = precisions_chol
        self._weights = estimator.weights_

    def predict_proba(self, X):
        #from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
        predictions = _estimate_log_gaussian_prob(X, self._means,
                                    self._precisions_col, self._covariance_type)

        predictions += numpy.log(self._weights)

        return predictions

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = numpy.argmax(probabilities, axis=1)
        return predictions

    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = "" # Implement
        if file:
            with open(file, 'w') as f:
                f.write(code)

        raise NotImplementedError("TODO implement save()")
        return code

