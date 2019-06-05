
from . import trees
from . import net
from . import bayes
from . import cgen

def convert(estimator, kind=None, method='pymodule', 
            float_type=None, float_precision=None):
    """Main entrypoint for converting a model"""


    if kind is None:
        kind = type(estimator).__name__
    
    if float_type is not None:
        cgen.FLOATING_DTYPE = float_type
    if float_precision is not None:
        cgen.FLOATING_PRECISION = float_precision
    # Uname instead of instance to avoid hard dependency on the libraries
    if kind in ['RandomForestClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']:
        return trees.Wrapper(estimator, method) 
    elif kind == 'MLPClassifier':
        return net.convert_sklearn_mlp(estimator, method)
    elif kind == 'Sequential':
        return net.convert_keras(estimator, method)
    elif kind == 'GaussianNB':
        return bayes.Wrapper(estimator, method)
    else:
        raise ValueError("Unknown model type: '{}'".format(kind))
