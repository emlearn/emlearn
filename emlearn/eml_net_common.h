
#ifndef EML_NET_COMMON_H
#define EML_NET_COMMON_H

/**
    Activation function. Used in layers
*/
typedef enum _EmlNetActivationFunction {
    EmlNetActivationIdentity = 0,
    EmlNetActivationRelu,
    EmlNetActivationLogistic,
    EmlNetActivationSoftmax,
    EmlNetActivationTanh,
    EmlNetActivationFunctions,
} EmlNetActivationFunction;

static const char *
eml_net_activation_function_strs[EmlNetActivationFunctions] = {
    "identity",
    "relu",
    "logistic",
    "softmax",
    "tanh",
};

#endif // EML_NET_COMMON_H
