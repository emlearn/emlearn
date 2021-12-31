
from . import trees
from . import net
from . import bayes
from . import distance
from . import mixture

def convert(estimator, kind=None, method='pymodule', dtype='float'):
    """Main entrypoint for converting a model"""

    if kind is None:
        kind = type(estimator).__name__

    # Use name instead of instance to avoid hard dependency on the libraries
    if kind in ['RandomForestClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']:
        return trees.Wrapper(estimator, method, dtype=dtype)
    elif kind in ['EllipticEnvelope']:
        return distance.Wrapper(estimator, method, dtype=dtype)
    elif kind in ['GaussianMixture', 'BayesianGaussianMixture']:
        return mixture.Wrapper(estimator, method, dtype=dtype)
    elif kind == 'MLPClassifier':
        return net.convert_sklearn_mlp(estimator, method)
    elif kind == 'Sequential':
        return net.convert_keras(estimator, method)
    elif kind == 'GaussianNB':
        return bayes.Wrapper(estimator, method)
    else:
        raise ValueError("Unknown model type: '{}'".format(kind))
