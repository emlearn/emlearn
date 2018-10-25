
import numpy

import eml_net

def convert_sklearn_mlp(model, method):

    if (model.n_layers_ < 3):
        raise ValueError("Model must have at least one hidden layer")

    weights = model.coefs_
    biases = model.intercepts_
    activations = [model.activation]*(len(weights)-1) + [ model.out_activation_ ]

    cmodel = eml_net.Classifier(activations, weights, biases)
    return cmodel

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
    for i, l in enumerate(model.layers):
        layer_type = type(l).__name__

        if layer_type == 'Activation':
            # merge dedicated Activation layers into the previous layer
            # TODO: maybe make activation a separate layer in our representation
            activations[-1] = from_keras_activation(l.activation)
            continue
        elif layer_type == 'Dense':
            assert l.use_bias == True, 'Layers without bias not supported'
            activations.append(from_keras_activation(l.activation))
            weights, bias  = l.get_weights()
            assert bias.ndim == 1, bias.ndim
            assert weights.ndim == 2, weights.ndim
            biases.append(bias)
            layer_weights.append(weights)
        else:
            raise NotImplementedError("Layer type '{}' is not implemented".format(layer_type)) 

    assert len(activations) == len(biases) == len(layer_weights)
    
    cmodel = eml_net.Classifier(activations, layer_weights, biases)
    return cmodel
