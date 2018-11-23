
from . import common

import eml_net

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
            self.classifier = eml_net.Classifier(activations, weights, biases)
        elif classifier == 'loadable':
            name = 'mynet'
            func = 'eml_net_predict_proba(&{}, values, length)'.format(name)
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

        code = c_generate_net(self.weights, self.activations, self.biases, name)
        if file:
            with open(file) as f:
                f.write(code)

        return code

def c_init(*args):
    start = '{'
    end = '}'
    return start + args.join(', ') + end

def c_generate_net(activations, weights, biases):
    def init_net(n_layers, layers_name, buf1_name, buf2_name, buf_length):
        return c_init(n_layers, layers_name, buf1_name, buf2_name, buf_length)
    def init_layer(n_outputs, n_inputs, weigths_name, biases_name, activation_func):
        return c_init(n_outputs, n_inputs, weights_name, biases_name, activation_func)

    nodes_structs = ',\n  '.join(node(n) for n in flat)
    nodes_name = name
    nodes_length = len(flat)
    nodes = "EmlTreesNode {nodes_name}[{nodes_length}] = {{\n  {nodes_structs} \n}};".format(**locals());

    return nodes

def convert_sklearn_mlp(model, method):

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

