
#include <stdio.h>

#include <eml_trees.h>
#include <eml_data.h>

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

};



void data_callback(EmlDataReader *reader, const unsigned char *buffer,
                    int length, int32_t item_no)
{
    if (item_no < 0) {
        fprintf(stderr, "ERROR: should not get header\n");  
        return;
    }

    py::array_t<int32_t> *arr = (py::array_t<int32_t> *)reader->user_data;

    const int ndims = eml_data_reader_ndims(reader);
    if (ndims == 2) {
        int x, y;
        eml_data_reader_2dcoord(reader, item_no, &x, &y);
        const int32_t val = eml_data_read_int32_le(buffer);
        //fprintf(stderr, "item %d: %d, %d: %d\n", item_no, x, y, val);  
        auto r = arr->mutable_unchecked<2>();
        r(x, y) = val;
    }
}

py::array_t<int32_t>
load_data(std::string input) {

    EmlDataReader reader;
    eml_data_reader_init(&reader);

    const char *buffer = input.c_str();
    size_t len = input.size();

    // read only the header
    eml_data_reader_chunk(&reader, buffer, EML_DATA_HEADER_LENGTH, 0);
    buffer += EML_DATA_HEADER_LENGTH;
    len -= EML_DATA_HEADER_LENGTH;
   

    if (reader.dtype != EmlDataInt32) {
        fprintf(stderr, "ERROR, unexpected datatype %d\n", reader.dtype);
    }

    // create output array
    auto out = py::array_t<int32_t>();
    const int ndims = eml_data_reader_ndims(&reader); 
    if (ndims == 2) {
        std::vector<int> shape = { reader.dim0, reader.dim1 };
        auto a = py::array_t<int32_t>(shape);
        out = a; 
    } else {
        fprintf(stderr, "ERROR: unsupported dimensions %d \n", ndims);  
    }

    // read rest of the data
    reader.user_data = (void *)&out;
    eml_data_reader_chunk(&reader, buffer, len, data_callback);


    return out;
}


PYBIND11_MODULE(eml_trees, m) {
    m.doc() = "Tree-based machine learning classifiers for embedded devices";

    py::class_<TreesClassifier>(m, "Classifier")
        .def(py::init<std::vector<float>, std::vector<int32_t>>())
        .def("predict", &TreesClassifier::predict);

    m.def("load_data", load_data);
}

