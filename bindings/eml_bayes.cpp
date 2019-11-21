
#include <stdio.h>
#include <stdlib.h>

#include <eml_bayes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Classifier {
private:
    EmlBayesSummary *summaries;
    EmlBayesModel model;

public:
    Classifier(std::vector<float> data, int n_classes, int n_features)
        : summaries(nullptr)
    {
        const int n_attributes = 3;
        // FIXME: check data is n_classes*n_features*n_attributes
        const int n_items = n_classes*n_features;
        summaries = (EmlBayesSummary *)malloc(sizeof(EmlBayesSummary)*n_items);

        model.n_features = n_features;
        model.n_classes = n_classes;
        model.summaries = summaries;

        for (int class_idx = 0; class_idx<n_classes; class_idx++) {
            for (int feature_idx = 0; feature_idx<n_features; feature_idx++) {
                const int32_t summary_idx = class_idx*n_features + feature_idx;
                EmlBayesSummary summary;
                summary.mean = EML_Q16_FROMFLOAT(data[n_attributes*summary_idx + 0]);
                summary.std = EML_Q16_FROMFLOAT(data[n_attributes*summary_idx + 1]);
                summary.stdlog2 = EML_Q16_FROMFLOAT(data[n_attributes*summary_idx + 2]);
                model.summaries[summary_idx] = summary;
            }
        }
    }
    ~Classifier() {
        free(summaries);
    }

    py::array_t<int32_t>
    predict(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = (int32_t)in.shape()[1];

        auto classes = py::array_t<int32_t>(n_samples);
        auto r = classes.mutable_unchecked<1>(); 
        for (int i=0; i<n_samples; i++) {
            const float *features = in.data(i);
            const int32_t p = eml_bayes_predict(&model, features, n_features);
            if (p < 0) {
                throw std::runtime_error("Unknown error");
            }
            r(i) = p;
        }

        return classes;
    }
};

float logpdf_float(float x, float mean, float std, float stdlog2) {
   return EML_Q16_TOFLOAT(eml_bayes_logpdf(EML_Q16_FROMFLOAT(x), EML_Q16_FROMFLOAT(mean), EML_Q16_FROMFLOAT(std), EML_Q16_FROMFLOAT(stdlog2)));
}

PYBIND11_MODULE(eml_bayes, m) {
    m.doc() = "Naive Bayes classifiers for microcontroller and embedded devices";

    // Probability function
    m.def("logpdf", logpdf_float);

    py::class_<Classifier>(m, "Classifier")
        .def(py::init<std::vector<float>, int, int>())
        .def("predict", &Classifier::predict);
}

