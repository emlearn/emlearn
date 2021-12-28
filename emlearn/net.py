
from . import common, cgen

import numpy

import os.path

def argmax(sequence):
    max_idx = 0
    max_value = sequence[0]
    for i, value in enumerate(sequence):
        if value > max_value:
            max_idx = i
            max_value = value
    return max_idx


class Wrapper:
    def __init__(self, activations, weights, biases, classifier):

        self.activations = activations
        self.weights = weights
        self.biases = biases
        self.classifier = None

        if classifier == 'pymodule':
            import eml_net # import when required
            self.classifier = eml_net.Classifier(activations, weights, biases)
        elif classifier == 'loadable':
            name = 'mynet'
            func = 'eml_net_predict(&{}, values, length)'.format(name)
            code = self.save(name=name)
            self.classifier = common.CompiledClassifier(code, name=name, call=func)
        #elif classifier == 'inline':
        else:
            raise ValueError("Unsupported classifier method '{}'".format(classifier))

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict(self, X):
        classes = self.classifier.predict(X)
        return classes

    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = c_generate_net(self.activations, self.weights, self.biases, name)
        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code


def c_generate_net(activations, weights, biases, prefix):
    def init_net(name, n_layers, layers_name, buf1_name, buf2_name, buf_length):
        init = cgen.struct_init(n_layers, layers_name, buf1_name, buf2_name, buf_length)
        o = 'static EmlNet {name} = {init};'.format(**locals())
        return o
    def init_layer(name, n_outputs, n_inputs, weigths_name, biases_name, activation_func):
        init = cgen.struct_init(n_outputs, n_inputs, weights_name, biases_name, activation_func)
        return init

    buffer_sizes = [ w.shape[0] for w in weights ] + [ w.shape[1] for w in weights ]
    buffer_size = max(buffer_sizes)
    n_layers = len(activations)

    layers_name = prefix+'_layers'
    buf1_name = prefix+'_buf1'
    buf2_name = prefix+'_buf2'

    head_lines = [
        '#include <eml_net.h>'    
    ]

    layer_lines = []
    layers = []
    for layer_no in range(0, n_layers):
        l_weights = weights[layer_no]
        l_bias = biases[layer_no]
        l_activation = activations[layer_no]

        n_in, n_out = l_weights.shape
        weights_name = '{prefix}_layer{layer_no}_weights'.format(**locals())
        biases_name = '{prefix}_layer{layer_no}_biases'.format(**locals())
        activation_func = 'EmlNetActivation'+l_activation.title()
        layer_name = '{prefix}_layer{layer_no}'.format(**locals())
    
        weight_values = numpy.array(l_weights).flatten(order='C')
        weights_arr = cgen.array_declare(weights_name, n_in * n_out, values=weight_values)
        layer_lines.append(weights_arr)
        bias_values = l_bias
        biases_arr = cgen.array_declare(biases_name, len(l_bias), values=bias_values)
        layer_lines.append(biases_arr)

        l = init_layer(layer_name, n_out, n_in, weights_name, biases_name, activation_func)
        layers.append('\n'+l)

    net_lines = [
        cgen.array_declare(buf1_name, buffer_size, modifiers='static'),
        cgen.array_declare(buf2_name, buffer_size, modifiers='static'),
        cgen.array_declare(layers_name, n_layers, dtype='EmlNetLayer', values=layers),
        init_net(prefix, n_layers, layers_name, buf1_name, buf2_name, buffer_size),
    ]

    name = prefix
    predict_function = f"""
    int32_t
    {name}_predict(const float *features, int32_t n_features)
    {{
        return eml_net_predict(&{name}, features, n_features);

    }}
    """

    lines = head_lines + layer_lines + net_lines + [predict_function]
    out = '\n'.join(lines)

    return out

def convert_sklearn_mlp(model, method):
    """Convert sklearn.neural_network.MLPClassifier models"""

    if (model.n_layers_ < 3):
        raise ValueError("Model must have at least one hidden layer")

    weights = model.coefs_
    biases = model.intercepts_
    activations = [model.activation]*(len(weights)-1) + [ model.out_activation_ ]

    return Wrapper(activations, weights, biases, classifier=method)

def from_keras_activation(act):
    name = act.__name__
    remap = {
        'sigmoid': 'logistic',
        'linear': 'identity',
    }
    return remap.get(name, name)

def from_tf_variable(var):
    array = var.eval()
    return array

def convert_keras(model, method):
    """Convert keras.Sequential models"""

    activations = []
    layer_weights = []
    biases = []

    def add_dense(activation, weights, bias):
        activations.append(from_keras_activation(l.activation))
        weights, bias  = l.get_weights()
        assert bias.ndim == 1, bias.ndim
        assert weights.ndim == 2, weights.ndim
        biases.append(bias)
        layer_weights.append(weights)

    def set_activation(activation):
        # merge dedicated Activation layers into the previous layer
        # TODO: maybe make activation a separate layer in our representation
        activations[-1] = activation

    for i, l in enumerate(model.layers):
        layer_type = type(l).__name__

        # Dense layers
        if layer_type == 'Dense':
            assert l.use_bias == True, 'Layers without bias not supported'
            add_dense(l.activation, *l.get_weights())
    
        # Activations
        elif layer_type == 'Activation':
            set_activation(from_keras_activation(l.activation))
            continue
        elif layer_type == 'ReLU':
            assert l.negative_slope == 0.0, 'ReLU.negative_slope must be 0.0'
            assert l.threshold == 0.0, 'ReLU.threshold must be 0.0'
            set_activation('relu')
            continue
        elif layer_type == 'Softmax':
            assert l.axis == -1, 'Softmax.axis must be -1'
            set_activation('softmax')
            continue

        # Training layers
        elif layer_type == 'Dropout':
            # only used at training time
            continue

        else:
            raise NotImplementedError("Layer type '{}' is not implemented".format(layer_type)) 

    assert len(activations) == len(biases) == len(layer_weights)
    
    return Wrapper(activations, layer_weights, biases, classifier=method)

