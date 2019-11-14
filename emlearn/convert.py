
from . import trees
from . import net
from . import bayes

def convert(estimator, kind=None, method='pymodule', dtype='float', compiler=None):
    """Main entrypoint for converting a model"""

    if kind is None:
        kind = type(estimator).__name__

    # Use name instead of instance to avoid hard dependency on the libraries
    if kind in ['RandomForestClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']:
        return trees.Wrapper(estimator, method, dtype=dtype, compiler=compiler)
    elif kind == 'MLPClassifier':
        return net.convert_sklearn_mlp(estimator, method, compiler=compiler)
    elif kind == 'Sequential':
        return net.convert_keras(estimator, method, compiler=compiler)
    elif kind == 'GaussianNB':
        return bayes.Wrapper(estimator, method, compiler=compiler)
    else:
        raise ValueError("Unknown model type: '{}'".format(kind))
