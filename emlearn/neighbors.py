
"""
Nearest Neighbors
=========================
"""

from . import common, cgen

import numpy

import os.path
import warnings

SUPPORTED_METRICS = set(['euclidean'])
SUPPORTED_WEIGHTS = set(['uniform'])

def check_params_supported(estimator):

    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(estimator)

    metric = estimator.effective_metric_
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f'Unsupported metric: {metric}. Supported: {SUPPORTED_METRICS}')

    weights = estimator.weights
    if weights not in SUPPORTED_WEIGHTS:
        raise ValueError(f'Unsupported weights: {weights}. Supported: {SUPPORTED_WEIGHTS}')

    algorithm = estimator.algorithm
    if algorithm not in ('brute', 'auto'):
        warnings.warn('emlearn only implements "brute" for Nearest Neighbors. Ignoring setting algorithm={algorithm}')


class Wrapper:
    def __init__(self, estimator, inference='loadable', return_type='classifier'):

        check_params_supported(estimator)

        self.fit_data_X = estimator._fit_X
        self.fit_data_Y = estimator._y
        self.n_neighbors = estimator.n_neighbors
        self.inference = inference

        name = 'mymodel'

        if inference == 'pymodule':
            raise NotImplementedError(f'inference=pymodule not yet supported')
            #import eml_net # import when required
            #self.classifier = eml_net.Classifier(activations, weights, biases)
        elif inference == 'loadable':
            distance_length = len(self.fit_data_X)
            n_features = self.fit_data_X.shape[1]

            model_init = self.save(name=name)

            code = '\n'.join([
                model_init,

                # Add a RAM array to store distances in, used internally during predict
                cgen.array_declare(name='distance_array', modifiers='static',
                    dtype='EmlNeighborsDistanceItem', size=distance_length),
                # Wrapper that is compatible with CompilerClassifier
                f"""
                int32_t
                predict_func(const float *values, int length) {{
                    int16_t out = -1;

                    // Convert to integer
                    int16_t features[{n_features}];
                    for (int i=0; i<length; i++) {{
                        features[i] = (int16_t)values[i];
                    }}
                    const EmlError err = \
                        eml_neighbors_predict(&{name}, features, length, distance_array, {distance_length}, &out);
                    if (err != EmlOk) {{
                        return -err;
                    }}
                    return out;
                }}
                """
            ])
            func = 'predict_func(values, length)'


            self.classifier = common.CompiledClassifier(code, name=name, call=func)
        else:
            raise ValueError(f"Unsupported inference method '{inference}'")

    def predict_proba(self, X):
        raise NotImplementedError('predict_proba() not yet supported')

    def predict(self, X):
        return self.classifier.predict(X)
 
    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = c_generate_neighbors(self.fit_data_X, n_neighbors=self.n_neighbors, labels=self.fit_data_Y, prefix=name)
        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code


def c_generate_convenience_functions(module_name, name,
        feature_type='int16_t',
        clasification_return_type='int16_t',
        ):

    predict_function = f"""
    {clasification_return_type}
    {name}_predict(const {feature_type} *features, int32_t n_features)
    {{
        return {module_name}_predict(&{name}, features, n_features);
    }}
    """
    # TODO: add regression wrappers

    return [ predict_function ]

def neighbors_model_init(name, n_neighbors, n_features, n_items, max_items, data, labels):

    # NOTE: order must match the EmlNeighbors C typedef
    values = ( n_features, n_items, max_items, data, labels, n_neighbors )
    out = cgen.struct_declare(name, type_name='EmlNeighborsModel', values=values)
    return out

def c_generate_neighbors(data, labels, n_neighbors, prefix,
            array_modifiers='static const'):

    cgen.assert_valid_identifier(prefix)

    model_name = prefix

    # X/data/features
    assert len(data.shape) == 2, data.shape
    data_name = prefix+'_data'
    n_items, n_features = data.shape
    data_values = data.flatten()

    # Y/labels
    assert len(labels.shape) == 1, labels.shape
    labels_name = prefix+'_labels'

    max_items = n_items

    head_lines = [
        '#include <eml_neighbors.h>'    
    ]

    def declare_array(name, values):
        return cgen.array_declare(name, values=values, dtype='int16_t', modifiers=array_modifiers)

    model_lines = [
        declare_array(data_name, values=data_values),
        declare_array(labels_name, values=labels),
        neighbors_model_init(name=model_name,
            labels=f'(int16_t *){labels_name}',
            data=f'(int16_t *){data_name}',
            n_neighbors=n_neighbors,
            n_features=n_features,
            n_items=n_items,
            max_items=max_items,
        ),
    ]

    #convenience_functions = c_generate_convenience_functions('eml_neighbors', name=prefix)
    
    lines = head_lines + model_lines
    out = '\n'.join(lines)

    return out

def convert_sklearn(model, inference):
    """Convert sklearn.neural_network.MLPClassifier models"""

    return Wrapper(model, inference=inference)


