
#include <stdio.h>

#include "emtrees.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

    int32_t predict(std::vector<EmtreesValue> values) {
        const int32_t p = emtrees_predict(&forest, &values[0], values.size());
        if (p < 0) {
            const std::string msg = emtrees_errors[-p];
            throw std::runtime_error(msg);
        }
        return p;
    }
};

PYBIND11_MODULE(emtreesc, m) {
    m.doc() = "Tree-based machine learning classifiers for embedded devices";

    py::class_<EmtreesClassifier>(m, "Classifier")
        .def(py::init<std::vector<EmtreesValue>, std::vector<int32_t>>())
        //.def_readwrite("dt", &PID::dt)
        .def("predict", &EmtreesClassifier::predict);
}

