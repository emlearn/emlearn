
"""
Tree evaluation metrics
=========================
Utilities for measuring a tree-based model
"""
import math

from ..convert import convert as convert_model
from ..trees import Wrapper as TreesWrapper

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
    if isinstance(model, TreesWrapper):
        em = model
    else:
        em = convert_model(model)
    
    nodes, roots, leaves = em.forest_
    return len(nodes)

def model_size_leaves(model, a=None, b=None):
    """
    """
    if isinstance(model, TreesWrapper):
        em = model
    else:
        em = convert_model(model)

    nodes, roots, leaves = em.forest_
    return len(leaves)

def model_size_bytes(model, a=None, b=None, node_size=None, leaf_size=None):
    """
    Size of model, in bytes. For both decision nodes and leaves
    """
    # EmlTreesNode is 56 bits
    # This is 8 bytes on most platforms due to padding/alignment
    # feature index, a threshold value, left, and right child indices
    if node_size is None:
        node_size = 8

    if isinstance(model, TreesWrapper):
        em = model
    else:
        em = convert_model(model)

    if leaf_size is None:
        leaf_size = math.ceil(em.leaf_bits/8)

    nodes = model_size_nodes(em)
    leaves = model_size_leaves(em)
    bytes = (nodes * node_size) + (leaves * leaf_size)

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
