
#include <stdio.h>

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define EML_DEBUG 1

#include "emlpy_common.hpp"
#include <eml_iir.h>

namespace py = pybind11;



py::array_t<float>
iirfilter_py(py::array_t<float, py::array::c_style | py::array::forcecast> sos,
          py::array_t<float, py::array::c_style | py::array::forcecast> data)

{
    EMLPY_PRECONDITION(sos.ndim() == 2, "SOS coefficients must have dimensions 2");
    EMLPY_PRECONDITION(data.ndim() == 1, "data must have dimensions 1");

    const int n_stages = (int)sos.shape(0);

    // Setup cascade
    std::vector<float> coefficients(sos.data(), sos.data() + 6*n_stages);
    std::vector<float> states(4*n_stages, 0.0);
    EmlIIR filter = {
        n_stages,
        states.data(),
        (int)states.size(),
        coefficients.data(),
        (int)coefficients.size(),
    };

    EMLPY_CHECK_ERROR(eml_iir_check(filter));

    // Storing output
    auto ret = py::array_t<float>(data.shape(0));
    float *retdata = (float *)ret.data();

    // Perform filter
    for (int i=0; i<data.shape(0); i++) {
        retdata[i] = eml_iir_filter(filter, data.data()[i]);
    }

    return ret;
}



PYBIND11_MODULE(eml_signal, m) {
    m.doc() = "Signal processing for emlearn";

    m.def("iirfilter", iirfilter_py);

}

