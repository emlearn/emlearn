
from . import trees
from . import net
from . import bayes

def convert(estimator, kind=None, method='pymodule'):
    if kind is None:
        kind = type(estimator).__name__

    if kind in ['RandomForestClassifier', 'ExtraTreesClassifier']:
        return trees.Wrapper(estimator, method) 
    elif kind == 'MLPClassifier':
        return net.convert_sklearn_mlp(estimator, method)
    elif kind == 'GaussianNB':
        return bayes.Wrapper(estimator, method)
    else:
        raise ValueError("Unknown model type: '{}'".format(kind))
