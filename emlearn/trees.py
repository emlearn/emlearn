
"""
Tree-based models
=========================
"""

import os.path
import os
import warnings
import math

import numpy

from . import common, cgen

SUPPORTED_ESTIMATORS=[
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'DecisionTreeClassifier',
    'RandomForestRegressor',
    'ExtraTreesRegressor',
    'DecisionTreeRegressor',
]


def quantize_probabilities_into_byte(p, bits=8):
    assert bits <= 8, bits
    assert bits >= 1, bits
    max = numpy.max(p)
    min = numpy.min(p)
    assert max <= 1.0, max
    assert min >= 0.0, min
    steps = (2**bits)-1

    # Quantize to n_bits levels
    quantized = numpy.round(p * (steps - 1))
    out_max = numpy.max(quantized)
    assert out_max <= (2**bits)-1, (out_max, (2**bits)-1)
    # Scale back to full uint8 range [0, 255]
    scaled = (quantized / (steps - 1) * 255).astype(numpy.uint8)
    return scaled

# Tree representation as 2 arrays
# array of decision nodes:
# DNODE: feature, value, left_child, right_child
# array of leaf nodes
# LEAF: 
def flatten_tree(tree, leaf='argmax', leaf_bits=8):
    decision_nodes = []
    leaf_nodes = []

    assert tree.node_count == len(tree.value)
    assert tree.value.shape[1] == 1 # number of outputs

    def add_leaf(idx):
        """
        Returns an updated index value to identify the leaf
        """
        value = tree.value[idx]
        assert len(value) == 1, 'only one output supported'

        if leaf == 'argmax':
            # majority voting
            val = numpy.argmax(value[0])
        elif leaf == 'value':
            # regression
            val = value[0][0]
        elif leaf == 'probabilities':
            val = quantize_probabilities_into_byte(value[0], bits=leaf_bits)

        leaf_data = val
        leaf_idx = len(leaf_nodes)
        leaf_nodes.append(leaf_data)
        encoded = -leaf_idx-1
        assert encoded <= -1 # 0 means decision node. So first leaf is -1
        return encoded

    def reference_node(idx):
        """
        Returns updated index value to identify decision node
        """
        
        n_leaves = len(leaf_nodes)
        decision_node_idx = idx - n_leaves
        #print('REF NODE', idx, decision_node_idx)
        assert decision_node_idx >= 0
        return decision_node_idx


    def process_child(idx): 
        is_leaf = tree.children_left[idx] == -1 and tree.children_right[idx] == -1
        if is_leaf:
            return add_leaf(idx)
        else:
            return idx # will be corrected later

    def add_decision_node(node):
        out_idx = len(decision_nodes)
        decision_nodes.append(node)
        decision_node_mapping[node_no] = out_idx

    decision_node_mapping = {}
    leaves_seen = 0
    zipped = list(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
    for node_no, (left, right, feature, th, value) in enumerate(zipped):

        if len(zipped) == 1:
            # single node tree - must be a leaf
            # this can happen in some edge cases, like min_samples_leaf is very high
            assert left == -1
            assert right == -1
            assert node_no == 0
            # add a dummy decision node, where both sides go to the leaf
            leaf_idx = add_leaf(0)
            add_decision_node([0, 0, leaf_idx, leaf_idx])
            break

        if left == -1 and right == -1:
            # is a leaf. Is handled via its parent
            leaves_seen += 1
            continue

        else:
            left = process_child(left)
            right = process_child(right)

            node = [ feature, th, left, right ]
            add_decision_node(node)
       
    # Update child decision node references to reflect smaller output nodes array
    for node in decision_nodes:
        if node[2] >= 0:
            node[2] = decision_node_mapping[node[2]]
        if node[3] >= 0:
            node[3] = decision_node_mapping[node[3]]


    total_nodes = len(decision_nodes) + len(leaf_nodes)
    if len(zipped) == 1:
        assert total_nodes == 2
        assert len(decision_nodes) == 1
        assert len(leaf_nodes) == 1
    else:
        assert total_nodes == tree.node_count, (total_nodes, tree.node_count)

    #print_tree((decision_nodes, leaf_nodes))

    assert_node_references_valid(decision_nodes, leaf_nodes, roots=[0])
    t = decision_nodes, leaf_nodes
    return t

def print_tree(tree):
    nodes, leaves = tree

    for i, n in enumerate(nodes):
        print('NODE', i, n)

    for i, n in enumerate(leaves):
        print('LEAF', i, n)

def print_forest(forest):
    nodes, roots, leaves = forest

    for i, n in enumerate(nodes):
        print('NODE', i, n)

    for i, n in enumerate(leaves):
        print('LEAF', i, n)

    for i, r in enumerate(roots):
        print('ROOT', i, r)

def assert_node_references_valid(nodes, leaves, roots):

    # INVARIANT. References to nodes in decision nodes are to valid nodes
    # TODO: check
    left_children = set([ n[2] for n in nodes if n[2] >= 0 ])
    right_children = set([ n[3] for n in nodes if n[3] >= 0 ])
    node_idxs = set(range(0, len(nodes)))

    invalid_children_left = left_children - node_idxs
    assert invalid_children_left == set(), invalid_children_left

    invalid_children_right = right_children - node_idxs
    assert invalid_children_right == set(), invalid_children_right

    extranous_nodes = node_idxs - (left_children | right_children | set(roots))
    assert extranous_nodes == set(), extranous_nodes

    # INVARIANT. References to leaves are to valid leaves
    left_leaves = set([ (-n[2])-1 for n in nodes if n[2] < 0 ])
    right_leaves = set([ (-n[3])-1 for n in nodes if n[3] < 0 ])
    leaf_idxs = set(range(0, len(leaves)))
    invalid_leaves_left = (left_leaves - leaf_idxs)
    assert invalid_leaves_left == set(), invalid_leaves_left

    invalid_leaves_right = (right_leaves - leaf_idxs)
    assert invalid_leaves_left == set(), invalid_leaves_left

    extranous_leaves = (leaf_idxs - (left_leaves | right_leaves))
    assert extranous_leaves == set(), extranous_leaves


def assert_forest_valid(forest):
    nodes, roots, leaves = forest

    assert_node_references_valid(nodes, leaves, roots)


def flatten_forest(trees, leaf='argmax', leaf_bits=0):
    tree_roots = []
    decision_nodes_offset = 0
    leaf_nodes_offset = 0
    forest_nodes = []
    forest_leaves = []

    for tree in trees: 
        decision_nodes, leaf_nodes = flatten_tree(tree, leaf=leaf, leaf_bits=leaf_bits)

        # Offset the nodes in tree, so they can be stored in one array 
        root = 0 + decision_nodes_offset
        for node in decision_nodes:
            if node[2] >= 0:
                node[2] += decision_nodes_offset
            else:
                node[2] -= leaf_nodes_offset
            if node[3] >= 0:
                node[3] += decision_nodes_offset
            else:
                node[3] -= leaf_nodes_offset
        decision_nodes_offset += len(decision_nodes)
        leaf_nodes_offset += len(leaf_nodes)

        tree_roots.append(root)
        forest_nodes += decision_nodes
        forest_leaves += leaf_nodes

        #print('offsets', decision_nodes_offset, leaf_nodes_offset)

        #print_forest((forest_nodes, tree_roots, forest_leaves))    
        assert_forest_valid((forest_nodes, tree_roots, forest_leaves))


    f = forest_nodes, tree_roots, forest_leaves
    #print_forest(f)    
    assert_forest_valid(f)
    return f


def remove_duplicate_leaves(forest):
    nodes, roots, leaves = forest

    def find_array_index(array_list, target):
       for i, arr in enumerate(array_list):
           if numpy.array_equal(target, arr):
               return i
       return None  # or None if not found

    # Determine de-duplicated leaves
    unique_leaves = []
    #unique_idx = []
    remap_leaves = {}
    for old_idx, node in enumerate(leaves):
        old_encoded = -old_idx-1
        found = find_array_index(unique_leaves, node)
        if found is None:
            new_idx = len(unique_leaves)
            unique_leaves.append(node)
            #unique_idx.append(old_encoded)
        else:
            new_idx = found # unique_idx[found]
            #encoded = -new_idx-1

        new_encoded = -new_idx-1
        remap_leaves[old_encoded] = new_encoded


    wasted_ratio = (len(leaves) - len(unique_leaves)) / len(nodes)
    
    # Update decision nodes to point to new leaves
    for n in nodes:
        n[2] = remap_leaves.get(n[2], n[2])
        n[3] = remap_leaves.get(n[3], n[3])

    f = nodes, roots, unique_leaves
    #print_forest(f)
    assert_forest_valid(f)
    return f


def assert_valid_child(value, child_max = 2**15, child_min = -2**15):
    assert value >= child_min, value
    assert value <= child_max, value

def encode_child(index, child):
    if child >= 0:
        # decision node, use relative jump
        assert child >= index
        encoded = child - index
        # must not become negative, would be confused with a leaf
        assert encoded >= 0
    else:
        # leaf node, leave as-as
        encoded = child

    assert_valid_child(encoded)
    return encoded

def generate_c_nodes(flat, name, dtype='float', modifiers='static const'):

    def make_node(index, node):
        feature, value, left_child, right_child = node

        left = encode_child(index, left_child)
        right = encode_child(index, right_child)
        value = cgen.constant(value, dtype=dtype)

        return "{{ {}, {}, {}, {} }}".format(feature, value, left, right)

    nodes_structs = ',\n  '.join(make_node(i, n) for i, n in enumerate(flat))
    nodes_name = name
    nodes_length = len(flat)
    nodes = "{modifiers} EmlTreesNode {nodes_name}[{nodes_length}] = {{\n  {nodes_structs} \n}};".format(**locals());

    out = nodes

    return out

def leaves_to_bytelist(leaves, leaf_bits):

    if leaf_bits == 0:
        return leaves

    elif leaf_bits == 32:
        arr = numpy.array(leaves).astype(numpy.float32)
        out = list(arr.tobytes())

        leaf_bytes = math.ceil(leaf_bits/8)
        expect_bytes = leaf_bytes*len(leaves)
        assert len(out) == expect_bytes, (len(out), expect_bytes) 
        return out
    elif leaf_bits <= 8:
        arr = numpy.array(leaves).astype(numpy.uint8)
        out = list(arr.tobytes())
        return out
    else: 
        raise ValueError(f"Unsupported number for leaf_bits: {leaf_bits}")
    

def generate_c_inlined(forest, name, n_features, n_classes=0, leaf_bits=0, dtype='float', classifier=True, include_proba=True, weight_modifiers='static const'):
    nodes, roots, leaves = forest

    cgen.assert_valid_identifier(name)

    tree_names = [ name + '_tree_{}'.format(i) for i,_ in enumerate(roots) ]

    ctype = dtype
    leaf_dtype = 'int'
    if not classifier:
        leaf_dtype = 'float'
    indent = 2

    leaves_dtype = 'uint8_t'
    leaves_array = leaves_to_bytelist(leaves, leaf_bits=leaf_bits)
    leaves_length = len(leaves_array)
    leaves_name = name+'_leaves';
    leaves_code = cgen.array_declare(leaves_name, leaves_length,
            modifiers=weight_modifiers, dtype=leaves_dtype, values=leaves_array)

    def c_leaf(data, depth):
        value = cgen.constant(data, dtype=leaf_dtype)
        return (depth*indent * ' ') + "return {};".format(value)
    def c_internal(n, depth):
        f = """{indent}if (features[{feature}] < {value}) {{
        {left}
        {indent}}} else {{
        {right}
        {indent}}}""".format(**{
            'feature': cgen.constant(n[0], dtype='int'),
            'value': cgen.constant(n[1], dtype=dtype),
            'left': c_node(n[2], depth+1),
            'right': c_node(n[3], depth+1),
            'indent': depth*indent*' ',
        })
        return f
    def c_node(idx, depth):
        if idx < 0:
            leaf_idx = -idx-1
            if not classifier:
                # regression, put value directly into leaf
                leaf_value = leaves[leaf_idx]
            elif leaf_bits == 0:
                # hard voting, put the class index directly into leaf
                leaf_value = leaves[leaf_idx]
            else:
                # soft voting, offet into a leaves array
                leaf_value = leaf_idx
            return c_leaf(leaf_value, depth+1)
        else:
            return c_internal(nodes[idx], depth+1)



    def tree_func(name, root, return_type='int32_t'):
        return """static inline {return_type} {function_name}(const {ctype} *features, int32_t features_length) {{
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

    def tree_vote_proba(name):
        return '_class = {}(features, features_length); out[_class] += 1.0f;'.format(name)

    def tree_vote_regressor(name):
        return 'avg += {}(features, features_length); '.format(name)

    forest_regressor_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{

        float avg = 0;

        {tree_predictions}
        
        return avg/{n_trees};
    }}
    """.format(**{
      'function_name': name+"_predict",
      'n_classes': n_classes,
      'n_trees': len(roots),
      'tree_predictions': '\n    '.join([ tree_vote_regressor(n) for n in tree_names ]),
      'ctype': ctype,
    })

    forest_predict_majority_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{

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
      'function_name': name+"_predict",
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_classifier(n) for n in tree_names ]),
      'ctype': ctype,
    })


    forest_proba_majority_func = """int {function_name}(const {ctype} *features, int32_t features_length, float *out, int out_length) {{

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {{
            out[i] = 0.0f;
        }}

        {tree_predictions}
    
        // compute mean
        for (int i=0; i<out_length; i++) {{
            out[i] = out[i] / {n_trees};
        }}
        return 0;
    }}
    """.format(**{
      'function_name': name+"_predict_proba",
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_proba(n) for n in tree_names ]),
      'n_trees': len(tree_names),
      'ctype': ctype,
    })



    # proportions
    def tree_vote_leaf_proportion(tree_no, name):
        leaves = leaves_name
        c = f"""offset = {n_classes}*{name}(features, features_length);
            for (int i=0; i<{n_classes}; i++) {{ out[i] += ({leaves}[offset+i]/255.0f); }}
        """
        return c
    forest_proba_proportions_func = """int {function_name}(const {ctype} *features, int32_t features_length, float *out, int out_length) {{

        int offset = 0;

        for (int i=0; i<out_length; i++) {{
            out[i] = 0.0f;
        }}

        {tree_predictions}
    
        // compute mean
        for (int i=0; i<out_length; i++) {{
            out[i] = out[i] / {n_trees};
        }}
        return 0;
    }}
    """.format(**{
      'function_name': name+"_predict_proba",
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_leaf_proportion(i, n) for i, n in enumerate(tree_names) ]),
      'n_trees': len(tree_names),
      'ctype': ctype,
    })


    forest_predict_proportions_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{

        float out[{n_classes}] = {{0.0f,}};

        int offset = 0;

        {tree_predictions}
    
        // argmax over probabilities
        int32_t most_voted_class = -1;
        float most_voted_proba = 0.0f;
        for (int32_t i=0; i<{n_classes}; i++) {{

            if (out[i] > most_voted_proba) {{
                most_voted_class = i;
                most_voted_proba = out[i];
            }}
        }}
        return most_voted_class;
    }}
    """.format(**{
      'function_name': name+"_predict",
      'n_classes': n_classes,
      'tree_predictions': '\n    '.join([ tree_vote_leaf_proportion(i, n) for i, n in enumerate(tree_names) ]),
      'ctype': ctype,
    })


    return_type = 'int32_t'
    forest_classifier_func = forest_predict_majority_func if leaf_bits == 0 else forest_predict_proportions_func
    forest_funcs = [forest_classifier_func]

    forest_proba_func = forest_proba_majority_func if leaf_bits == 0 else forest_proba_proportions_func
    if include_proba:
        forest_funcs += [forest_proba_func]

    if not classifier:
        return_type = 'float'
        forest_funcs = [ forest_regressor_func ]

    tree_funcs = [tree_func(n, r, return_type=return_type) for n,r in zip(tree_names, roots)]

    head = """
    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    """

    parts  = [head] + tree_funcs
    if leaf_bits != 0:
        parts += [ leaves_code ]
    parts += forest_funcs
    out = '\n\n'.join(parts)

    return out


def generate_c_loadable(forest, name, n_features,
        weight_modifiers='static const', dtype='float',
        classifier=True, n_classes=0, leaf_bits=0, include_proba=True):

    nodes, roots, leaves = forest

    cgen.assert_valid_identifier(name)

    nodes_name = name+'_nodes'
    nodes_length = len(nodes)
    nodes_c = generate_c_nodes(nodes, nodes_name, dtype=dtype, modifiers=weight_modifiers)

    tree_roots_length = len(roots)
    tree_roots_name = name+'_tree_roots';
    tree_roots_values = ', '.join(str(t) for t in roots)
    tree_roots = '{weight_modifiers} int32_t {tree_roots_name}[{tree_roots_length}] = {{ {tree_roots_values} }};'.format(**locals())

    leaves_dtype = 'uint8_t'
    leaves_array = leaves_to_bytelist(leaves, leaf_bits=leaf_bits)
    leaves_length = len(leaves_array)
    leaves_name = name+'_leaves';
    leaves = cgen.array_declare(leaves_name, leaves_length,
            modifiers=weight_modifiers, dtype=leaves_dtype, values=leaves_array)
   
    # The inference strategy uses either 0, 32 or 8 (for quantizations 1-7)
    tree_leaf_bits = leaf_bits if leaf_bits == 0 or leaf_bits == 32 else 8

    forest_struct = """EmlTrees {name} = {{
        {nodes_length},
        (EmlTreesNode *)({nodes_name}),	  
        {tree_roots_length},
        (int32_t *)({tree_roots_name}),
        {leaves_length},
        ({leaves_dtype} *)({leaves_name}),
        {tree_leaf_bits},
        {n_features},
        {n_classes},
    }};""".format(**locals())


    # Provide convenience wrapper functions, to not have to deal with the underlying structs etc
    ctype = dtype

    forest_regressor_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{

        const float out = eml_trees_regress1(&{model_name}, features, features_length);
        return out;
    }}
    """.format(**{
      'model_name': name,
      'function_name': name+"_predict",
      'n_trees': len(roots),
      'ctype': ctype,
    })

    forest_classifier_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{

        const int out = eml_trees_predict(&{model_name}, features, features_length);
        return out;

    }}
    """.format(**{
      'model_name': name,
      'function_name': name+"_predict",
      'ctype': ctype,
    })


    forest_proba_func = """int {function_name}(const {ctype} *features, int32_t features_length, float *out, int out_length) {{

        const EmlError err = \
            eml_trees_predict_proba(&{model_name}, features, features_length, out, out_length);
        return err;

    }}
    """.format(**{
      'model_name': name,
      'function_name': name+"_predict_proba",
      'ctype': ctype,
    })

    forest_funcs = [forest_classifier_func]
    if include_proba:
        forest_funcs += [forest_proba_func]

    if not classifier:
        return_type = 'float'
        forest_funcs = [ forest_regressor_func ]


    head = """
    // !!! This file is generated using emlearn !!!

    #include <eml_trees.h>
    """

    code = '\n\n'.join([head, nodes_c, tree_roots, leaves, forest_struct] + forest_funcs)
    return code


class Wrapper:
    def __init__(self, estimator, method, dtype='int16_t', leaf_bits=None):

        if method is None:
            method = 'inline'

        self.dtype = dtype
        if self.dtype is None:
            self.dtype = 'int16_t'

        kind = type(estimator).__name__
        leaf = 'argmax'
        self.is_classifier = True
        self.out_dtype = "int"
        if 'Regressor' in kind:
            leaf = 'value'
            self.is_classifier = False
            self.out_dtype = "float"

        if leaf_bits == 1:
            # treat as majority voting
            leaf_bits = 0

        if leaf_bits is None:
            if self.is_classifier:
                leaf_bits = 0
            else:
                leaf_bits = 32

        if leaf_bits > 0 and leaf_bits <= 8 and self.is_classifier:
            leaf = 'probabilities'

        self.leaf_bits = leaf_bits

        if hasattr(estimator, 'estimators_'):
            estimators = [ e for e in estimator.estimators_]
        else:
            estimators = [ estimator ]

        trees = [ e.tree_ for e in estimators ]

        self.forest_ = flatten_forest(trees, leaf=leaf, leaf_bits=self.leaf_bits)
        self.forest_ = remove_duplicate_leaves(self.forest_)


        self.n_features = estimators[0].n_features_in_
        self.n_classes = 0
        if self.is_classifier:
            self.n_classes = estimators[0].n_classes_
        self.method = method
        if self.method not in ('loadable', 'inline'):
            raise ValueError("Unsupported inference method '{}'".format(self.method))

        if self.method == 'loadable' and self.dtype != 'int16_t':
            raise ValueError("Inference method='loadable' only supports dtype='int16_t'. Use method='inline' for others")

        # TODO: support more features for inline. Like 255
        max_features = 127 if self.method == 'loadable' else 10000
        if self.n_features > max_features:
            raise ValueError(f"Maximum features exceeded. features={self.n_features} max={max_features}")

        self._classifier = None # lazy-initialized by _build_classifier

    def _build_classifier(self):
        if self._classifier is not None:
            return None

        name = 'mytree'
        n_features = self.n_features
        n_classes = self.n_classes
        feature_dtype = self.dtype

        model_init = self.save(name=name, include_proba=True)

        return_type = 'int32_t' if self.is_classifier else 'float'

        # Floating point wrappers that are compatible with CompilerClassifier
        classifier_functions = [
            f"""
            {return_type}
            predict_wrapper(const float *values, int length) {{
                // Convert to whatever is needed for inline
                {feature_dtype} features[{n_features}];
                for (int i=0; i<length; i++) {{
                    features[i] = ({feature_dtype})values[i];
                }}
                const {return_type} out = {name}_predict(features, length);
                if (out < 0) {{
                    return -out;
                }}
                return out;
            }}""",
            f"""
            int
            predict_proba_wrapper(const float *values, int length, float *outputs, int n_outputs) {{
                // Convert to whatever is needed for inline
                {feature_dtype} features[{n_features}];
                for (int i=0; i<length; i++) {{
                    features[i] = ({feature_dtype})values[i];
                }}

                const int err = \
                    {name}_predict_proba(features, length, outputs, n_outputs);

                return err;
            }}
            """,
        ]

        regression_functions = [
            f"""
            {return_type}
            regress_wrapper(const float *values, int length) {{
                // Convert to whatever is needed for inline
                {feature_dtype} features[{n_features}];
                for (int i=0; i<length; i++) {{
                    features[i] = ({feature_dtype})values[i];
                }}
                const {return_type} out = {name}_predict(features, length);
                return out;
            }}
            """,
        ]

        sections = [model_init]
        if self.is_classifier:
            sections += classifier_functions
        else:
            sections += regression_functions

        code = '\n'.join(sections)

        predict_func = 'predict_wrapper(values, length)'
        regress_func = 'regress_wrapper(values, length)'
        proba_func = 'predict_proba_wrapper(values, length, outputs, N_CLASSES)'

        if self.is_classifier:
            call_func = predict_func
        else:
            call_func = regress_func
            proba_func = None

        self.classifier_ = common.CompiledClassifier(code, name=name,
            call=call_func, proba_call=proba_func,
            out_dtype=self.out_dtype, n_classes=self.n_classes,
        )


    def predict(self, X):
        self._build_classifier()

        if self.is_classifier:
            predictions = self.classifier_.predict(X)
        else:
            predictions = self.classifier_.regress(X)            

        return predictions

    def predict_proba(self, X):
        self._build_classifier()

        if not self.is_classifier:
            raise ValueError(f"Cannot call predict_proba on a Regressor")
        
        probabilities = self.classifier_.predict_proba(X)
        return probabilities

    def save(self, name=None, file=None, format='c', inference=None, include_proba=True):

        if inference is None:
            inference = [self.method]
        else:
            if len(inference) != 1:
                raise ValueError('Only support specifying on inference type on save')
            warnings.warn("The 'inference' argument is deprecated. It will be removed in a future version. Use method= in constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        if format == 'c':
            code = ""
            generate_args = dict(forest=self.forest_,
                name=name,
                dtype=self.dtype,
                classifier=self.is_classifier,
                leaf_bits=self.leaf_bits,
                n_classes=self.n_classes,
                n_features=self.n_features,
                include_proba=include_proba,
            )
            if 'loadable' in inference:
                code += '\n\n' + generate_c_loadable(**generate_args)
            if 'inline' in inference:
                code += '\n\n' + generate_c_inlined(**generate_args)
            if not code:
                raise ValueError("No code generated. Check that 'inference' specifies valid strategies")

        elif format == 'csv':
            nodes, roots, leaves = self.forest_
            nodes = nodes.copy()
            lines = []

            lines.append(f'f,{self.n_features}')
            lines.append(f'c,{self.n_classes}')
            for l in leaves:
                lines.append(f'l,{l}')
            for r in roots:
                lines.append(f'r,{r}')

            def serialize_node(index, node):
                feature, value, left_child, right_child = node

                left = encode_child(index, left_child)
                right = encode_child(index, right_child)

                serialized = f'n,{feature},{value.round(6)},{left},{right}'
                return serialized

            for i, n in enumerate(nodes):
                lines.append(serialize_node(i, n))

            code = '\r\n'.join(lines) 
        else:
            raise ValueError(f"Unsupported format: {format}")

        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code


