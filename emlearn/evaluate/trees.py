
"""
Tree evaluation metrics
=========================
Convert a Python model into C code
"""

from ..convert import convert as convert_model

import numpy

def get_tree_estimators(estimator):
    """
    Get the DecisionTree instances from ensembles or single-tree models
    """
    if hasattr(estimator, 'estimators_'):
        trees = [ e for e in estimator.estimators_]
    else:
        trees = [ estimator ]
    return trees
        
def model_size_nodes(model, a=None, b=None):    
    """
    Size of model, in number of decision nodes
    """
    em = convert_model(model)
    
    nodes, roots = em.forest_
    return len(nodes)

def model_size_bytes(model, a=None, b=None, node_size=None):
    """
    Size of model, in bytes
    """
    # EmlTreesNode consists of feature index, a threshold value, left-right child indices 
    if node_size is None:
        node_size = 1+4+2+2

    nodes = model_size_nodes(model)
    bytes = nodes * node_size
    return bytes

def tree_depth_average(model, a=None, b=None):
    """
    Average depth of model
    """
    trees = get_tree_estimators(model)
    depths = [ e.tree_.max_depth for e in trees ]
    return numpy.mean(depths)

def tree_depth_difference(model, a=None, b=None):
    """Measures how much variation there is in tree depths"""
    trees = get_tree_estimators(model)
    depths = [ e.tree_.max_depth for e in trees ]
    return numpy.max(depths) - numpy.min(depths)

def count_trees(model, a=None, b=None):
    """
    Number of trees in model
    """
    trees = get_tree_estimators(model)
    return len(trees)

def compute_cost_estimate(model, X, b=None):
    """
    Make an estimate of the compute cost, using the following assumptions:
    
    - The dataset X is representative of the typical dataset
    - Cost is proportional to the number of decision node evaluation in a tree
    - The cost is added across all trees in the ensemble

    Under this model, the actual compute time can be computed as the estimate times a constant C,
    representing the time a single evaluation of a decision node takes.
    """
    trees = get_tree_estimators(model)
    
    X = numpy.array(X)

    total = 0.0
    for e in trees:
        path = e.decision_path(X)
        t = numpy.sum(path, axis=1)
        total += numpy.mean(t)
    
    return total
