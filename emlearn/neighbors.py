
"""
Nearest Neighbors
=========================
"""

from . import common, cgen

import numpy

import os.path

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

    print(dir(estimator))
    # TODO: emit warning if algorithm is not ('brute', 'auto')


class Wrapper:
    def __init__(self, estimator, inference='loadable', return_type='classifier'):

        check_params_supported(estimator)


        self.fit_data = estimator._fit_X
        self.inference = inference

        name = 'mymodel'

        if inference == 'pymodule':
            raise NotImplementedError(f'inference=pymodule not yet supported')
            #import eml_net # import when required
            #self.classifier = eml_net.Classifier(activations, weights, biases)
        elif inference == 'loadable':
            func = 'eml_neighbors_predict(&{}, values, length)'.format(name)
            code = self.save(name=name)
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

        code = c_generate_neighbors(self.fit_data, prefix=name)
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

def c_generate_neighbors(data, prefix,
            array_modifiers='static const'):

    cgen.assert_valid_identifier(prefix)

    data_name = prefix+'_data'
    data_size = 100
    labels_name = prefix+'_labels'

    head_lines = [
        '#include <eml_neighbors.h>'    
    ]

    model_lines = [
        cgen.array_declare(data_name, data_size, modifiers=array_modifiers),
        #cgen.array_declare(labels_name, buffer_size, modifiers='static'),
        #cgen.array_declare(layers_name, n_layers, dtype='EmlNetLayer', values=layers),
        #init_neighbors(prefix, ),
    ]

    convenience_functions = c_generate_convenience_functions('eml_neighbors', name=prefix)
    
    lines = head_lines + model_lines + convenience_functions
    out = '\n'.join(lines)

    return out

def convert_sklearn(model, inference):
    """Convert sklearn.neural_network.MLPClassifier models"""

    return Wrapper(model, inference=inference)


