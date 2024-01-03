
"""
Neural networks
=========================
"""

from . import common, cgen
from .fixedpoint import FixedPointFormat

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
    def __init__(self, activations, weights, biases, classifier,
            return_type='classifier',
            use_fixedpoint=False,
        ):

        self.activations = activations
        self.weights = weights
        self.biases = biases
        self.classifier = None
        self.return_type = return_type
        self.inference_type = classifier
        self.use_fixedpoint = use_fixedpoint

        if self.use_fixedpoint and self.inference_type != 'inline':
            raise NotImplementedError("Fixed-point only implemented with 'inline' inference type")

        if self.inference_type == 'pymodule' and return_type == 'classifier':
            import eml_net # import when required
            self.classifier = eml_net.Classifier(activations, weights, biases)
        elif self.inference_type == 'loadable' and return_type == 'classifier':
            name = 'mynet'
            func = 'eml_net_predict(&{}, values, length)'.format(name)
            code = self.save(name=name)
            self.classifier = common.CompiledClassifier(code, name=name, call=func)
        elif self.inference_type == 'loadable' and return_type == 'regressor':
            name = 'mynet'
            func = 'eml_net_regress1(&{}, values, length)'.format(name)
            code = self.save(name=name)
            self.classifier = common.CompiledClassifier(code, name=name, call=func, out_dtype='float')
        elif self.inference_type == 'inline' and return_type == 'classifier':
            name = 'mynet'
            code = self.save(name=name)

            if self.use_fixedpoint:
                # inject a conversion between float and fixed-point
                n_features = self.weights[0].shape[0]
                code += f"""
                eml_q16_t {name}_fixed_values[{n_features}];

                int {name}_predict_float(const float *values, int length) {{
                    for (int i=0; i<length; i++) {{
                        {name}_fixed_values[i] = EML_Q16_FROMFLOAT(values[i]);
                    }}
                    return {name}_predict({name}_fixed_values, length); 
                }}
                """
                func = f'{name}_predict_float(values, length)'                
            else:
                func = f'{name}_predict(values, length)'
            self.classifier = common.CompiledClassifier(code, name=name, call=func)
        elif self.inference_type == 'inline' and return_type == 'regressor':
            raise NotImplementedError("Inline inference not supported for regressors, use loadable instead")
        else:
            raise ValueError(f"Unsupported classifier method '{classifier}' with return_type of '{return_type}'")

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict(self, X):
        if self.return_type == 'classifier':
            return self.classifier.predict(X)
        elif self.return_type == 'regressor':
            return self.classifier.regress(X)
        else:
            raise ValueError(f"Unsupported return_type of '{self.return_type}'")
 

    def save(self, name=None, file=None, inference=['loadable']):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        if ('loadable' in inference) and ('inline' in inference):
            raise ValueError("Specify either 'loadable' or 'inline' inference. Both together is not supported")

        code = ""
        if 'loadable' in inference:
            code += '\n' + c_generate_net_loadable(self.activations, self.weights, self.biases, prefix=name)
        if 'inline' in inference:
            code += '\n' + c_generate_net_inline(self.activations, self.weights, self.biases,
                prefix=name,
                use_fixedpoint=self.use_fixedpoint,
            )
        if not code:
            raise ValueError("No code generated. Check that 'inference' specifies valid strategies")

        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code

def array_declare(name, fixedpoint : FixedPointFormat = None, **kwargs):

    # if fixedpoint==None, uses float
    return cgen.array_declare_fixedpoint(name, fixedpoint=fixedpoint, **kwargs)


def c_generate_layer_data(activations, weights, biases, prefix : str,
            include_constants=True,
            use_fixedpoint=False,
            arr_modifiers = 'static const'):

    declarations = []
    def add_declaration(code):
        declarations.append(dict(code=code))
    def format_name(layer_no, variable):
        name = f'{prefix}_layer_{layer_no}_{variable}'
        return name

    # TODO: pick most appropriate fixed-point format
    weights_format = FixedPointFormat(integer_bits=8, fraction_bits=23) if use_fixedpoint else None

    # Layers
    for layer_no, (l_act, l_weights, l_bias) in enumerate(zip(activations, weights, biases)):
        n_in, n_out = l_weights.shape

        # layer sizes
        if include_constants:
            in_name = format_name(layer_no, 'input_length')
            add_declaration(cgen.constant_declare(in_name, n_in))
            out_name = format_name(layer_no, 'output_length')
            add_declaration(cgen.constant_declare(out_name, n_out))

        # activation
        if include_constants:
            activation_name = format_name(layer_no, 'activation')
            activation_func = 'EmlNetActivation'+l_act.title()
            add_declaration(cgen.constant_declare(activation_name, activation_func))

        # bias
        biases_name = format_name(layer_no, 'biases') 
        biases_arr = array_declare(biases_name, size=len(l_bias),
            values=l_bias, modifiers=arr_modifiers, fixedpoint=weights_format)
        add_declaration(biases_arr)

        # weights
        weights_name = format_name(layer_no, 'weights') 
        weight_values = numpy.array(l_weights).flatten(order='C')
        weights_arr = array_declare(weights_name, size=n_in * n_out,
            values=weight_values, modifiers=arr_modifiers, fixedpoint=weights_format)
        add_declaration(weights_arr)

    return declarations

def c_generate_net_inline(activations, weights, biases, prefix : str,
        use_fixedpoint = False,
        data_modifiers : str = 'static const'):
    """
    Generate C code for a particular neural network. Aka the "inline" inference strategy
    """

    cgen.assert_valid_identifier(prefix)

    arr_modifiers = data_modifiers
    buffer_modifiers = 'static'

    buffers_ctype = 'eml_fixed32_t' if use_fixedpoint else 'float'
    template_name = "net_fixedpoint.jinja" if use_fixedpoint else "net_float.jinja" 

    # Load template
    from jinja2 import Environment, FileSystemLoader
    here = os.path.dirname(__file__)
    template_dir = os.path.join(here, "templates/")
    environment = Environment(loader=FileSystemLoader(template_dir))
    template = environment.get_template(template_name)

    # Generate declarations
    declarations = []
    def add_declaration(code):
        declarations.append(dict(code=code))
    def format_name(layer_no, variable):
        name = f'{prefix}_layer_{layer_no}_{variable}'
        return name

    # Working buffers
    buffer_sizes = [ w.shape[0] for w in weights ] + [ w.shape[1] for w in weights ]
    buffer_size = max(buffer_sizes)
    add_declaration(cgen.constant_declare(f'{prefix}_activations_length', buffer_size))
    add_declaration(cgen.array_declare(f'{prefix}_activations1', dtype=buffers_ctype, modifiers=buffer_modifiers, size=buffer_size))
    add_declaration(cgen.array_declare(f'{prefix}_activations2', dtype=buffers_ctype, modifiers=buffer_modifiers, size=buffer_size))

    # Number of outputs
    n_outputs = weights[-1].shape[1]
    add_declaration(cgen.constant_declare(f'{prefix}_n_outputs', n_outputs))

    # Layers
    declarations += c_generate_layer_data(activations, weights, biases, prefix, use_fixedpoint=use_fixedpoint)

    # Generate the neural network code
    layer_numbers = list(range(len(activations)))

    out = template.render(
        prefix=prefix,
        declarations=declarations,
        layers=layer_numbers,
    )

    return out


def c_generate_net_loadable(activations, weights, biases, prefix):
    """
    Generate general C code for neural networks inference. Aka the "loadable" inference strategy
    """

    def init_net(name, n_layers, layers_name, buf1_name, buf2_name, buf_length):
        init = cgen.struct_init(n_layers, layers_name, buf1_name, buf2_name, buf_length)
        o = 'static EmlNet {name} = {init};'.format(**locals())
        return o
    def init_layer(name, n_outputs, n_inputs, weights_name, biases_name, activation_func):
        init = cgen.struct_init(n_outputs, n_inputs, weights_name, biases_name, activation_func)
        return init

    cgen.assert_valid_identifier(prefix)

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

    layer_declarations = c_generate_layer_data(activations, weights, biases, prefix,
            include_constants=False)
    for d in layer_declarations:
        layer_lines.append(d['code'])

    for layer_no, (l_act, l_weights) in enumerate(zip(activations, weights)):
        n_in, n_out = l_weights.shape
        layer = f'{prefix}_layer_{layer_no}'

        activation_func = 'EmlNetActivation'+l_act.title()
        l = init_layer(layer, n_out, n_in, f'{layer}_weights', f'{layer}_biases', activation_func)
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

    regress_function = f"""
    int32_t
    {name}_regress(const float *features, int32_t n_features, float *out, int32_t out_length)
    {{
        return eml_net_regress(&{name}, features, n_features, out, out_length);
    }}
    """

    regress1_function = f"""
    float
    {name}_regress1(const float *features, int32_t n_features)
    {{
        return eml_net_regress1(&{name}, features, n_features);
    }}
    """

    lines = head_lines + layer_lines + net_lines + [predict_function, regress_function, regress1_function]
    out = '\n'.join(lines)

    return out

def convert_sklearn_mlp(model, method, **kwargs):
    """Convert sklearn.neural_network.MLPClassifier models"""

    if (model.n_layers_ < 3):
        raise ValueError("Model must have at least one hidden layer")

    weights = model.coefs_
    biases = model.intercepts_
    activations = [model.activation]*(len(weights)-1) + [ model.out_activation_ ]

    return Wrapper(activations, weights, biases, classifier=method, **kwargs)

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

def convert_keras(model, method, **kwargs):
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
    
    return Wrapper(activations, layer_weights, biases, classifier=method, **kwargs)

