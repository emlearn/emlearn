
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
    std::vector<uint8_t> leaves;
public:
    TreesClassifier(std::vector<float> node_data,
            std::vector<int32_t> _roots,
            std::vector<uint8_t> leaves,
            int leaf_bits,
            int n_classes,
            int n_features)
        : roots(_roots)
    {
        // TODO: take model coefficients as a Numpy array (perf) 
        // FIXME: check node_data is multiple of 4

        /* Decision nodes */
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

        /* Tree roots*/
        forest.n_trees = roots.size();
        forest.tree_roots = &roots[0];

        /* Leaves */
        this->leaves = leaves;
        forest.leaf_bits = leaf_bits;
        forest.leaves = &this->leaves[0];
        forest.n_leaves = this->leaves.size();

        /* Metadata */
        forest.n_classes = n_classes;
        forest.n_features = n_features;
    }
    ~TreesClassifier() {
        // FIXME: free leaves
        free(nodes);
    }

    py::array_t<float>
    predict_proba(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 2) {
            throw std::runtime_error("predict input must have dimensions 2");
        }

        const int64_t n_samples = in.shape()[0];
        const int32_t n_features = in.shape()[1];
        const int n_classes = this->forest.n_classes;
       
        auto probabilities = py::array_t<float>(std::vector<ptrdiff_t>{n_samples, n_classes});
        
        for (int i=0; i<n_samples; i++) {
            const float *v = in.data(i);
            float *p = probabilities.mutable_data(i);
            const EmlError err = eml_trees_predict_proba(&forest, v, n_features, p, n_classes);
            if (err != EmlOk) {
                const std::string msg = eml_error_str(err);
                throw std::runtime_error(msg);
            }
        }

        return probabilities;
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
        .def(py::init<std::vector<float>,
            std::vector<int32_t>,
            std::vector<uint8_t>,
            int,
            int,
            int>())
        .def("regress", &TreesClassifier::regress)
        .def("predict_proba", &TreesClassifier::predict_proba)
        .def("predict", &TreesClassifier::predict);
}

