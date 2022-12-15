
#include <stdio.h>

#include <eml_trees.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class TreesClassifier {
private:
    std::vector<int32_t> roots;
    EmlTreesNode *nodes;
    EmlTrees forest;

public:
    TreesClassifier(std::vector<float> node_data, std::vector<int32_t> _roots)
        : roots(_roots)
    {
        // TODO: take model coefficients as a Numpy array (perf) 
        // FIXME: check node_data is multiple of 4
        const int n_nodes = node_data.size() / 4;
        nodes = (EmlTreesNode *)malloc(sizeof(EmlTreesNode)*n_nodes);

        for (int i=0; i<n_nodes; i++) {
            EmlTreesNode n = {
                (int8_t)node_data[i*4+0],
                node_data[i*4+1],
                (int16_t)node_data[i*4+2],
                (int16_t)node_data[i*4+3]
            };
            nodes[i] = n;
        }
    
        forest.nodes = nodes;
        forest.n_nodes = n_nodes;
        forest.n_trees = roots.size();
        forest.tree_roots = &roots[0];
    }
    ~TreesClassifier() {
        free(nodes);
    }


    py::array_t<int32_t>
    predict(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = in.shape()[1];

        auto classes = py::array_t<int32_t>(n_samples);
        //auto s = in.unchecked();
        auto r = classes.mutable_unchecked<1>(); 
        for (int i=0; i<n_samples; i++) {
            const float *v = in.data(i);
            const int32_t p = eml_trees_predict(&forest, v, n_features);
            if (p < 0) {
                const std::string msg = eml_trees_errors[-p];
                throw std::runtime_error(msg);
            }
            r(i) = p;
        }

        return classes;
    }

    py::array_t<int32_t>
    regress(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = in.shape()[1];

        auto outputs = py::array_t<float>(n_samples);
        //auto s = in.unchecked();
        auto r = outputs.mutable_unchecked<1>();
        float out[1];
        for (int i=0; i<n_samples; i++) {
            const float *v = in.data(i);
            const EmlError err = eml_trees_regress(&forest, v, n_features, out, 1);
            if (err != EmlOk) {
                const std::string msg = eml_error_str(err);
                throw std::runtime_error(msg);
            }
            r(i) = out[0];
        }

        return outputs;
    }

};

PYBIND11_MODULE(eml_trees, m) {
    m.doc() = "Tree-based machine learning classifiers for embedded devices";

    py::class_<TreesClassifier>(m, "Classifier")
        .def(py::init<std::vector<float>, std::vector<int32_t>>())
        .def("regress", &TreesClassifier::regress)
        .def("predict", &TreesClassifier::predict);
}

