
from . import trees

def convert(estimator, kind=None, method='pymodule'):
    if kind is None:
        kind = type(estimator).__name__

    if kind in ['RandomForestClassifier', 'ExtraTreesClassifier']:
        return trees.Wrapper(estimator, method) 
    else:
        raise ValueError("Unknown model type: '{}'".format(kind))
