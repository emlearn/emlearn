
#include <stdio.h>

#include <eml_net.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Returns an EmlNetActivationFunction or -EmlError
int32_t
eml_net_activation_func(const char *str)
{
    int32_t ret = -EmlUnsupported;

    for (int i=0; i<EmlNetActivationFunctions; i++) {
        const char *func_str = eml_net_activation_function_strs[i];
        if (strcmp(str, func_str) == 0) {
            ret = (int32_t)i;
        }
    }

    return ret;
}

typedef py::array_t<float, py::array::c_style | py::array::forcecast> FloatArray;

class EmlNetClassifier {
private:
    std::vector<int32_t> roots;
    std::vector<EmlNetLayer> layers;
    std::vector<FloatArray> weights;
    std::vector<FloatArray> biases;
    std::vector<float> activations_buffer_1;
    std::vector<float> activations_buffer_2;
    EmlNet model = {0,};

public:
    EmlNetClassifier(std::vector<std::string> activations,
                std::vector<FloatArray> _weights,
                std::vector<FloatArray> _biases)

    {
        // store reference
        weights = _weights;
        biases = _biases;

        // Layers
        if (weights.size() < 2) {
            throw std::runtime_error("Must have at least 3 layers total (1 hidden)");
        }

        model.n_layers = (int32_t)weights.size();
        layers = std::vector<EmlNetLayer>(model.n_layers);
        model.layers = layers.data();

        for (int i=0; i<model.n_layers; i++) {
            const int32_t a = eml_net_activation_func(activations[i].c_str());
            if (a < 0) {
                throw std::runtime_error("Unsupported activation function: " + activations[i]);
            }

            layers[i].n_inputs = (int32_t)weights[i].shape(0); 
            layers[i].n_outputs = (int32_t)weights[i].shape(1);
            layers[i].activation = (EmlNetActivationFunction)a;
            layers[i].weights = (float *)weights[i].data();
            layers[i].biases = (float *)biases[i].data();
        }

        // Buffers for activations
        const int32_t act_max = eml_net_find_largest_layer(&model);
        activations_buffer_1 = std::vector<float>(act_max);
        activations_buffer_2 = std::vector<float>(act_max);
        model.activations1 = activations_buffer_1.data();
        model.activations2 = activations_buffer_2.data();
        model.activations_length = act_max;
    }

    ~EmlNetClassifier() {

    }

    py::array_t<int32_t>
    predict(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = (int32_t)in.shape()[1];

        auto classes = py::array_t<int32_t>(n_samples);
        //auto s = in.unchecked();
        auto r = classes.mutable_unchecked<1>(); 
        for (int i=0; i<n_samples; i++) {
            const float *v = in.data(i);
            const int32_t p = eml_net_predict(&model, v, n_features);
            if (p < 0) {
                const std::string err = eml_error_str((EmlError)-p);
                throw std::runtime_error("Prediction error: " + err);
            }
            r(i) = p;
        }

        return classes;
    }

    py::array_t<float>
    predict_proba(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = (int32_t)in.shape()[1];
        const int32_t n_outputs = eml_net_outputs_proba(&model);

        const auto out_shape = std::vector<int64_t>{n_samples, n_outputs};

        auto proba = py::array_t<float>(out_shape);

        for (int i=0; i<n_samples; i++) {
            const float *v = in.data(i);
            float *out = (float *)proba.data(i);
            const EmlError e = eml_net_predict_proba(&model, v, n_features, out, n_outputs);
            if (e != EmlOk) {
                throw std::runtime_error("Prediction error: " + std::string(eml_error_str(e)));
            }
        }

        return proba;
    }

};

PYBIND11_MODULE(eml_net, m) {
    m.doc() = "Neural networks for embedded devices";

    py::class_<EmlNetClassifier>(m, "Classifier")
        .def(py::init< std::vector<std::string>, std::vector<FloatArray>, std::vector<FloatArray> >())
        .def("predict", &EmlNetClassifier::predict)
        .def("predict_proba", &EmlNetClassifier::predict_proba);
}

