
"""
Model conversion
=========================
Convert a Python model into C code
"""

from . import trees
from . import net
from . import bayes
from . import distance
from . import mixture

class Model():
    """Inference model powered by emlearn

    Wrapper around the underlying C code.
    Allows using the model in Python code, for evaluation/testing.
    Follows scikit-learn API conventions.
    """

    def __init__(self):
        pass

    def save(self, name : str, file : str = None) -> str:
        """Generate C code

        :param name: Name for the model. Must be valid C identifier
        :param file: Filepath for where to save the C code (optional)

        :return: The generated C code
        """

    def predict(self, X):
        """Run inference, return classes / regression values"""
        pass
    def predict_proba(self, X):
        """Run inference, return probabilities"""
        pass
    def score_samples(self, X):
        """Run inference, return anomaly/outlier scores"""
        pass


def convert(estimator, 
        kind : str = None,
        method: str = 'pymodule',
        dtype: str ='float',
        ) -> Model:
    """Convert model to C

    :param method: The inference strategy to use. pymodule|inline|loadable
    :param dtype: Datatype to use for features. Can be used to enable quantization 
    :param kind: Explicit name for the type of model. Useful if the model is a subclass of a supported model class
    :return: A Estimator like class, that uses C code for inference
    """

    if kind is None:
        kind = type(estimator).__name__

    # Use name instead of instance to avoid hard dependency on the libraries
    if kind in set(trees.SUPPORTED_ESTIMATORS):
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
