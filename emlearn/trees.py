

import os.path
import os

import numpy

from . import common

SUPPORTED_ESTIMATORS=[
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'DecisionTreeClassifier',
    'RandomForestRegressor',
    'ExtraTreesRegressor',
    'DecisionTreeRegressor',
]

# Tree representation as 2d array
# feature, value, left_child, right_child
# Leaf node: -1, class, -1, -1
# 	fields = { 'feature': 0, 'value': 1, 'left': 2, 'right': 3 }
def flatten_tree(tree, leaf='argmax'):
    flat = []

    assert tree.node_count == len(tree.value)
    assert tree.value.shape[1] == 1 # number of outputs

    for left, right, feature, th, value in zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value):
        if left == -1 and right == -1:
            if leaf == 'argmax':
                val = numpy.argmax(value[0])
            elif leaf == 'value':
                val = value[0][0]

            n = [ -1, val, -1, -1 ] # leaf
        else:
            n = [ feature, th, left, right ]

        flat.append(n) 

    return flat


def flatten_forest(trees, leaf='argmax'):
    tree_roots = []
    tree_offset = 0
    forest_nodes = []

    for tree in trees: 
        flat = flatten_tree(tree, leaf=leaf)

        # Offset the nodes in tree, so they can be stored in one array 
        root = 0 + tree_offset
        for node in flat:
            if node[2] > 0:
                node[2] += tree_offset
            if node[3] > 0:
                node[3] += tree_offset
        tree_offset += len(flat)
        tree_roots.append(root)
        forest_nodes += flat

    return forest_nodes, tree_roots

def remap_node_references(nodes, remap):
    for n in nodes:
        n[2] = remap.get(n[2], n[2])
        n[3] = remap.get(n[3], n[3])

def remove_orphans(nodes, roots):
    referenced = []
    for n in nodes:
        if n[0] >= 0:
            referenced.append(n[2])
            referenced.append(n[3])
    referenced = set(referenced).union(roots)
    all_nodes = set(range(len(nodes))) 
    orphaned = all_nodes.difference(referenced)


    offsets = []	
    offset = 0
    for idx, node in enumerate(nodes):
        offsets.append(offset)
        if idx in orphaned:
            offset -= 1

    compacted = []
    for idx, node in enumerate(nodes):
        if idx in orphaned:
            continue

        if node[0] >= 0:
            node[2] += offsets[node[2]]
            node[3] += offsets[node[3]] 
        compacted.append(node)

    compacted_roots = [ r + offsets[r] for r in roots ]

    return compacted, compacted_roots


def remove_duplicate_leaves(forest):
    nodes, roots = forest

    unique_leaves = []
    unique_idx = []
    remap_leaves = {}
    for i, node in enumerate(nodes):
        if node[0] >= 0:
            # not a leaf
            continue
        found = unique_leaves.index(node) if node in unique_leaves else None
        if found is None:
            unique_leaves.append(node)
            unique_idx.append(i)
        else:
            remap_leaves[i] = unique_idx[found]	

    leaves = list(filter(lambda n: n[0] < 0, nodes))
    wasted = (len(leaves) - len(unique_leaves)) / len(nodes)
    
    remap_node_references(nodes, remap_leaves)
    compacted, compacted_roots = remove_orphans(nodes, roots)

    return compacted, compacted_roots

def traverse_dfs(nodes, idx, visitor):
    visitor(idx)
    if nodes[idx][0] < 0:
        return None
    traverse_dfs(nodes, nodes[idx][2], visitor)
    traverse_dfs(nodes, nodes[idx][3], visitor)

def dot_node(name, **opts):
    return '{name} [label={label}];'.format(name=name, label=opts['label'])
def dot_edge(src, tgt, **opts):
    return '{src} -> {tgt} [taillabel={label}, labelfontsize={f}];'.format(src=src,tgt=tgt,label=opts['label'], f=opts['labelfontsize'])
def dot_cluster(name, nodes, indent='  '):
    name = 'cluster_' + name
    n = ('\n'+indent).join(nodes)
    return 'subgraph {name} {{\n  {nodes}\n}}'.format(name=name, nodes=n)

def forest_to_dot(forest, name='trees', indent="  "):
    nodes, roots = forest

    leaf_nodes = list(filter(lambda i: nodes[i][0] < 0, range(len(nodes))))
    trees = [ [] for r in roots ]
    for tree_idx, root in enumerate(roots):
        collect = []
        traverse_dfs(nodes, root, lambda i: collect.append(i))
        trees[tree_idx] = set(collect).difference(leaf_nodes)

    edges = []
    leaves = []
    clusters = []

    # group trees using cluster
    for tree_idx, trees in enumerate(trees):
        decisions = []
        for idx in trees:
            node = nodes[idx]
            n = dot_node(idx, label='"{}: feature[{}] < {}"'.format(idx, node[0], node[1]))
            left = dot_edge(idx, node[2], label='"  1"', labelfontsize=8)
            right = dot_edge(idx, node[3], label='"  0"', labelfontsize=8)
            decisions += [ n ]
            edges += [ left, right]

        clusters.append(dot_cluster('_tree_{}'.format(tree_idx), decisions, indent=2*indent))

    # leaves shared between trees
    for idx in leaf_nodes:
        node = nodes[idx]
        leaves += [ dot_node(idx, label='"{}"'.format(node[1])) ]

    dot_items = clusters + edges + leaves

    graph_options = {
        #'rankdir': 'LR',
        #'ranksep': 0.07,
    }

    variables = {
        'name': name,
        'options': ('\n'+indent).join('{}={};'.format(k,v) for k,v in graph_options.items()),
        'items': ('\n'+indent).join(dot_items),
    }
    dot = """digraph {name} {{
      // Graph options
      {options}

      // Nodes/edges
      {items}
    }}""".format(**variables)

    return dot


def generate_c_nodes(flat, name):
    def node(n):
        return "{{ {}, {}, {}, {} }}".format(*n)

    nodes_structs = ',\n  '.join(node(n) for n in flat)
    nodes_name = name
    nodes_length = len(flat)
    nodes = "EmlTreesNode {nodes_name}[{nodes_length}] = {{\n  {nodes_structs} \n}};".format(**locals());

    out = nodes

    return out

def generate_c_inlined(forest, name, dtype='float', classifier=True):
    nodes, roots = forest

    def is_leaf(n):
      return n[0] < 0
    def class_value(n):
      assert is_leaf(n)
      return n[1]

    if classifier:
        class_values = set(map(class_value, filter(is_leaf, nodes)))
        assert min(class_values) == 0
        n_classes = max(class_values)+1
    else:
        n_classes = numpy.shape(forest[1])[0]

    tree_names = [ name + '_tree_{}'.format(i) for i,_ in enumerate(roots) ]

    ctype = dtype

    indent = 2
    def c_leaf(n, depth):
        return (depth*indent * ' ') + "return {};".format(n[1])
    def c_internal(n, depth):
        f = """{indent}if (features[{feature}] < {value}) {{
        {left}
        {indent}}} else {{
        {right}
        {indent}}}""".format(**{
            'feature': n[0],
            'value': n[1],
            'left': c_node(n[2], depth+1),
            'right': c_node(n[3], depth+1),
            'indent': depth*indent*' ',
        })
        return f
    def c_node(nid, depth):
        n = nodes[nid]
        if n[0] < 0:
            return c_leaf(n, depth+1)
        return c_internal(n, depth+1)

    def tree_func(name, root, return_type='int32_t'):
        return """static inline int32_t {function_name}(const {ctype} *features, int32_t features_length) {{
        {code}
        }}
        """.format(**{
            'function_name': name,
            'code': c_node(root, 0),
            'ctype': ctype,
            'return_type': return_type 
        })

    def tree_vote_classifier(name):
        return '_class = {}(features, features_length); votes[_class] += 1;'.format(name)

    def tree_vote_regressor(name):
        return 'avg += {}(features, features_length); '.format(name)

    forest_regressor_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{

        float avg = 0;

        {tree_predictions}
        
        return avg/{n_classes};
    }}
    """.format(**{
      'function_name': name,
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_regressor(n) for n in tree_names ]),
      'ctype': ctype,
    })

    forest_classifier_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{

        int32_t votes[{n_classes}] = {{0,}};
        int32_t _class = -1;

        {tree_predictions}
    
        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<{n_classes}; i++) {{

            if (votes[i] > most_voted_votes) {{
                most_voted_class = i;
                most_voted_votes = votes[i];
            }}
        }}
        return most_voted_class;
    }}
    """.format(**{
      'function_name': name,
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_classifier(n) for n in tree_names ]),
      'ctype': ctype,
    })

    return_type = 'int32_t'
    forest_func = forest_classifier_func
    
    if not classifier:
        return_type = 'float'
        forest_func = forest_regressor_func

    tree_funcs = [tree_func(n, r, return_type=return_type) for n,r in zip(tree_names, roots)]

    return '\n\n'.join(tree_funcs + [forest_func])

def generate_c_forest(forest, name='myclassifier', dtype='float', classifier=True):
    nodes, roots = forest

    nodes_name = name+'_nodes'
    nodes_length = len(nodes)
    nodes_c = generate_c_nodes(nodes, nodes_name)

    tree_roots_length = len(roots)
    tree_roots_name = name+'_tree_roots';
    tree_roots_values = ', '.join(str(t) for t in roots)
    tree_roots = 'int32_t {tree_roots_name}[{tree_roots_length}] = {{ {tree_roots_values} }};'.format(**locals())

    forest_struct = """EmlTrees {name} = {{
        {nodes_length},
        {nodes_name},	  
        {tree_roots_length},
        {tree_roots_name},
    }};""".format(**locals())

    head = """
    // !!! This file is generated using emlearn !!!

    #include <eml_trees.h>
    """

    inline = generate_c_inlined(forest, name+'_predict', dtype=dtype, classifier=classifier)

    return '\n\n'.join([head, nodes_c, tree_roots, forest_struct, inline]) 




class Wrapper:
    def __init__(self, estimator, classifier, dtype='float'):

        self.dtype = dtype

        kind = type(estimator).__name__
        leaf = 'argmax'
        self.is_classifier = True
        out_dtype = "int"
        if 'Regressor' in kind:
            leaf = 'value'
            self.is_classifier = False
            out_dtype = "float"

        if hasattr(estimator, 'estimators_'):
            trees = [ e.tree_ for e in estimator.estimators_]
        else:
            trees = [ estimator.tree_ ]

        self.forest_ = flatten_forest(trees, leaf=leaf)
        self.forest_ = remove_duplicate_leaves(self.forest_)

        if classifier == 'pymodule':
            # FIXME: use Nodes,Roots directly, as Numpy Array
            import eml_trees # import when required
            nodes, roots = self.forest_
            node_data = []
            for node in nodes:
                assert len(node) == 4
                node_data += node
            assert len(node_data) % 4 == 0

            self.classifier_ = eml_trees.Classifier(node_data, roots)

        elif classifier == 'loadable':
            name = 'mytree'

            if self.is_classifier:
                func = 'eml_trees_predict(&{}, values, length)'.format(name)
            else:
                func = 'eml_trees_regress1(&{}, values, length)'.format(name)
            code = self.save(name=name)
            self.classifier_ = common.CompiledClassifier(code, name=name, call=func, out_dtype=out_dtype)
        elif classifier == 'inline':
            name = 'myinlinetree'
            func = '{}_predict(values, length)'.format(name)
            code = self.save(name=name)
            self.classifier_ = common.CompiledClassifier(code, name=name, call=func, out_dtype=out_dtype)
        else:
            raise ValueError("Unsupported classifier method '{}'".format(classifier))

    def predict(self, X):
        if self.is_classifier:
            predictions = self.classifier_.predict(X)
        else:
            predictions = self.classifier_.regress(X)            

        return predictions

    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = generate_c_forest(self.forest_, name, dtype=self.dtype, classifier=self.is_classifier)
        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code

    def to_dot(self, **kwargs):
        return forest_to_dot(self.forest_, **kwargs)


