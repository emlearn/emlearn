
#include <stdio.h>

#include <eml_trees.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class EmtreesClassifier {
private:
    std::vector<int32_t> roots;
    EmtreesNode *nodes;
    Emtrees forest;

public:
    EmtreesClassifier(std::vector<EmtreesValue> node_data, std::vector<int32_t> _roots)
        : roots(_roots)
    {
        // TODO: take model coefficients as a Numpy array (perf) 
        // FIXME: check node_data is multiple of 4
        const int n_nodes = node_data.size() / 4;
        nodes = (EmtreesNode *)malloc(sizeof(EmtreesNode)*n_nodes);

        for (int i=0; i<n_nodes; i++) {
            EmtreesNode n = {
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
    ~EmtreesClassifier() {
        free(nodes);
    }


    py::array_t<int32_t>
    predict(py::array_t<int32_t, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = in.shape()[1];

        auto classes = py::array_t<int32_t>(n_samples);
        //auto s = in.unchecked();
        auto r = classes.mutable_unchecked<1>(); 
        for (int i=0; i<n_samples; i++) {
            //const int32_t *v = s.data(i);
            const int32_t *v = in.data(i);
            const int32_t p = emtrees_predict(&forest, v, n_features);
            if (p < 0) {
                const std::string msg = emtrees_errors[-p];
                throw std::runtime_error(msg);
            }
            r(i) = p;
        }

        return classes;
    }

};

PYBIND11_MODULE(emtreesc, m) {
    m.doc() = "Tree-based machine learning classifiers for embedded devices";

    py::class_<EmtreesClassifier>(m, "Classifier")
        .def(py::init<std::vector<EmtreesValue>, std::vector<int32_t>>())
        .def("predict", &EmtreesClassifier::predict);
}

