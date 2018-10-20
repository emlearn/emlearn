
import eml_net

def convert_sklearn_mlp(model, method):

    if (model.n_layers_ < 3):
        raise ValueError("Model must have at least one hidden layer")

    weights = model.coefs_
    biases = model.intercepts_
    activations = [model.activation]*(len(weights)-1) + [ model.out_activation_ ]

    cmodel = eml_net.Classifier(activations, weights, biases)
    return cmodel

