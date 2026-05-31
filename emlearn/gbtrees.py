"""
Gradient Boosted Trees
======================

Support for GradientBoostingClassifier inference on embedded devices.
Binary classification only in this initial version.
"""

from .trees import assert_node_references_valid
from . import cgen
import os
from . import common
import numpy

def flatten_tree(tree):
    """Flatten a tree into a list of nodes, for easier code generation

    :param tree: The tree to flatten

    :return: two lists as arrays: nodes and leaves values. 
    """

    decision_nodes = []
    leaf_nodes = []



    def add_leaf(idx):
        

        value = tree.value[idx]
        assert len(value) == 1, 'only one output supported'
    
        
        leaf_data = value[0][0]
        leaf_idx = len(leaf_nodes)
        leaf_nodes.append(leaf_data)
        encoded = -leaf_idx-1
        assert encoded <= -1 # 0 means decision node. So first leaf is -1
        return encoded

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

def generate_c_inline(nodes, leaves, name, dtype='float'):
    
    leaf_dtype = 'float'
    indent = 2
    ctype = dtype

    def c_leaf(data, depth):
        value = cgen.constant(data, dtype=leaf_dtype)
        return (depth*indent * ' ') + "return {};".format(value)

    def c_internal(node, depth):
        f = """{indent}if (features[{feature}] < {value}) {{
        {left}
        {indent}}} else {{
        {right}
        {indent}}}""".format(**{
            'feature': cgen.constant(node[0], dtype='int'),
            'value': cgen.constant(node[1], dtype=dtype),
            'left': c_node(node[2], depth+1),
            'right': c_node(node[3], depth+1),
            'indent': depth*indent*' ',
        })
        return f

    def c_node(idx, depth):       
        if idx < 0:
            leaf_idx = -idx - 1
            leaf_value = leaves[leaf_idx]
            return c_leaf(leaf_value, depth + 1)
        else:
            return c_internal(nodes[idx], depth + 1)

    return """static inline float {name}(const {ctype} *features, int32_t features_length) {{
        {code}
    }}""".format(
        name=name,
        ctype=ctype,
        code=c_node(0, 0),
    )

def generate_c_trees(trees_per_class, name):
    c_functions = []
    for k, trees in enumerate(trees_per_class):
        for stage, tree in enumerate(trees):
            nodes, leaves = flatten_tree(tree)
            c_code = generate_c_inline(nodes, leaves, name=f"{name}_tree_stage{stage}_class{k}")
            c_functions.append(c_code)
    return c_functions

    
def generate_c_predict_proba(trees_per_class, learning_rate, init_score, name, dtype='float'):
    n_classes = len(trees_per_class)
    n_stages = len(trees_per_class[0])
    
    if n_classes == 1:
        # binary classification - sigmoid
        def c_tree_call(stage):
            return f"{name}_tree_stage{stage}_class0(features, features_length)"

        tree_calls = " + ".join([c_tree_call(stage) for stage in range(n_stages)])
        out = """int {name}_predict_proba(const float *features, int32_t features_length, float *out, int out_length) {{
        float raw = {init_score};
        raw += {learning_rate}*({tree_calls});
        float p1 = 1.0f / (1.0f + expf(-raw));
        out[0] = 1.0f - p1;
        out[1] = p1;
        return 0;
        }}""".format(
                name=name,
                learning_rate=cgen.constant(learning_rate, dtype=dtype),
                init_score=cgen.constant(init_score, dtype=dtype),
                tree_calls=tree_calls,
            )
        
    else:
        # multiclass - softmax
        # Todo: multi-class softmax Fall 
        pass

    return out

def generate_c_gbtrees_classifier(trees_per_class, learning_rate, init_score, name, dtype='float', **kwargs):
    header = "#include <stdint.h>\n#include <math.h>\n"
    tree_funcs = generate_c_trees(trees_per_class, name)
    proba_func = generate_c_predict_proba(trees_per_class, learning_rate, init_score, name, dtype)
    predict_func = generate_c_predict(trees_per_class, name)
    return header + "\n\n" + "\n\n".join(tree_funcs) + "\n\n" + proba_func + "\n\n" + predict_func


def generate_c_predict(trees_per_class, name):
    n_classes = len(trees_per_class)
    return """int32_t {name}_predict(const float *features, int32_t features_length) {{
    float out[{n_classes}];
    {name}_predict_proba(features, features_length, out, {n_classes});
    int32_t best = 0;
    for (int32_t i = 1; i < {n_classes}; i++) {{
        if (out[i] > out[best]) best = i;
    }}
    return best;
}}""".format(name=name, n_classes=2)  # binary: immer 2


def _extract_sklearn_gbc_trees(estimator):
    trees_per_class = []
    n_stages, n_classes_internal = estimator.estimators_.shape
    for k in range(n_classes_internal):
        trees = [estimator.estimators_[stage, k].tree_ for stage in range(n_stages)]
        trees_per_class.append(trees)

    lr = estimator.learning_rate
    init_score = float(estimator._raw_predict_init(numpy.zeros((1, estimator.n_features_in_)))[0][0])
    n_classes_internal = 1 if estimator.n_classes_ == 2 else estimator.n_classes_

    return trees_per_class, lr, init_score, n_classes_internal

class Wrapper:
    """
    emlearn wrapper for Gradient Boosted Tree models.
 
    Supports sklearn.ensemble.GradientBoostingClassifier and
    sklearn.ensemble.GradientBoostingRegressor.
    """
 
    def __init__(self, estimator, method=None, dtype='float'):
        if method is None:
            method = 'inline'
 
        if method != 'inline':
            raise ValueError(
                f"gbtrees only supports method='inline' (got '{method}')"
            )
 
        self.method = method
        self.dtype = dtype if dtype is not None else 'float'
 
        kind = type(estimator).__name__
        self.is_classifier = 'Classifier' in kind
 
        if self.is_classifier:
            self._init_classifier(estimator)
        else:
            self._init_regressor(estimator)
 
        self._classifier = None  # lazy init


 
    def _init_classifier(self, estimator):
        kind = type(estimator).__name__
 
        if kind == 'HistGradientBoostingClassifier':
            raise NotImplementedError(
                "HistGradientBoostingClassifier is not yet supported. "
                "Use GradientBoostingClassifier instead."
            )
 
        trees_per_class, lr, init_score, n_classes_internal = \
            _extract_sklearn_gbc_trees(estimator)
 
        self._trees_per_class = trees_per_class
        self._learning_rate = lr
        self._init_score = init_score
        self._n_classes_internal = n_classes_internal  # 1 for binary
        self.n_classes = estimator.n_classes_
        self.n_features = estimator.n_features_in_
 
    def _init_regressor(self, estimator):
        trees, lr, init_score = _extract_sklearn_gbr_trees(estimator)
        self._trees = trees
        self._learning_rate = lr
        self._init_score = init_score
        self.n_features = estimator.n_features_in_
        self.n_classes = 0
 
    def save(self, name=None, file=None, format='c', **kwargs):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            name = os.path.splitext(os.path.basename(file))[0]
 
        if format != 'c':
            raise ValueError(f"gbtrees only supports format='c' (got '{format}')")
 
        if self.is_classifier:
            code = generate_c_gbtrees_classifier(
                trees_per_class=self._trees_per_class,
                learning_rate=self._learning_rate,
                init_score=self._init_score,
                n_classes_internal=self._n_classes_internal,
                n_classes_out=self.n_classes,
                n_features=self.n_features,
                name=name,
                dtype=self.dtype,
            )
        else:
            code = generate_c_gbtrees_regressor(
                trees=self._trees,
                learning_rate=self._learning_rate,
                init_score=self._init_score,
                n_features=self.n_features,
                name=name,
                dtype=self.dtype,
            )
 
        if file:
            with open(file, 'w') as f:
                f.write(code)
 
        return code
 
    def _build_classifier(self):
        if self._classifier is not None:
            return
 
        name = 'mygbtree'
        n_features = self.n_features
        n_classes = self.n_classes
        feature_dtype = self.dtype
 
        model_code = self.save(name=name)
 
        if self.is_classifier:
            wrapper_funcs = f"""
int32_t predict_wrapper(const float *values, int length) {{
    {feature_dtype} features[{n_features}];
    for (int i = 0; i < length; i++) {{
        features[i] = ({feature_dtype})values[i];
    }}
    return {name}_predict(features, length);
}}
 
int predict_proba_wrapper(const float *values, int length, float *outputs, int n_outputs) {{
    {feature_dtype} features[{n_features}];
    for (int i = 0; i < length; i++) {{
        features[i] = ({feature_dtype})values[i];
    }}
    return {name}_predict_proba(features, length, outputs, n_outputs);
}}
"""
            code = model_code + "\n" + wrapper_funcs
            self._classifier = common.CompiledClassifier(
                code, name=name,
                call='predict_wrapper(values, length)',
                proba_call='predict_proba_wrapper(values, length, outputs, N_CLASSES)',
                out_dtype='int',
                n_classes=n_classes,
            )
        else:
            wrapper_func = f"""
float regress_wrapper(const float *values, int length) {{
    {feature_dtype} features[{n_features}];
    for (int i = 0; i < length; i++) {{
        features[i] = ({feature_dtype})values[i];
    }}
    return {name}_predict(features, length);
}}
"""
            code = model_code + "\n" + wrapper_func
            self._classifier = common.CompiledClassifier(
                code, name=name,
                call='regress_wrapper(values, length)',
                proba_call=None,
                out_dtype='float',
                n_classes=0,
            )
 
    def predict(self, X):
        self._build_classifier()
        if self.is_classifier:
            return self._classifier.predict(X)
        else:
            return self._classifier.regress(X)
 
    def predict_proba(self, X):
        self._build_classifier()
        if not self.is_classifier:
            raise ValueError("Cannot call predict_proba on a regressor")
        return self._classifier.predict_proba(X)