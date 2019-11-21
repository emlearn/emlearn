
#include <stdio.h>

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define EML_DEBUG 1

#include <eml_fft.h>
#include <eml_audio.h>

#include "emlpy_common.hpp"

namespace py = pybind11;


py::array_t<float>
rfft_py(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
    if (in.ndim() != 1) {
        throw std::runtime_error("SFT input must have dimensions 1");
    }

    const int n_fft = (int)in.shape(0);

    // Precompute FFT table
    const int n_fft_table = n_fft/2;
    std::vector<float> fft_sin(n_fft_table);
    std::vector<float> fft_cos(n_fft_table);
    EmlFFT fft = { n_fft_table, fft_sin.data(), fft_cos.data() };
    eml_fft_fill(fft, n_fft);

    // Setup working buffers
    const float * in_data = (const float *)in.data();
    std::vector<float> imag(n_fft);
    std::vector<float> real(n_fft);
    for (size_t i=0; i<real.size(); i++) {
        real[i] = in_data[i];
    }
    std::fill(imag.begin(), imag.end(), 0);

    // Do FFT
    const int status = eml_fft_forward(fft, real.data(), imag.data(), n_fft);

    if (status != EmlOk) {
        throw std::runtime_error("eml_fft error");
    }

    // Copy to output
    auto ret = py::array_t<float>(n_fft);
    for (size_t i=0; i<real.size(); i++) {
        ((float *)ret.data())[i] = real[i];
    }

    return ret;
}


py::array_t<float>
melfilter_py(py::array_t<float, py::array::c_style | py::array::forcecast> in,
    int samplerate, int n_fft, int n_mels, float fmin, float fmax
)
{
    if (in.ndim() != 1) {
        throw std::runtime_error("melfilter() input must have dimensions 1");
    }
    if (in.shape(0) != 1+(n_fft/2)) {
        throw std::runtime_error("framing not implented");
    }

    const EmlAudioMel params = { n_mels, fmin, fmax, n_fft, samplerate };

    // Copy input to avoid modifying
    const int length = (int)in.shape(0);
    EmlVector inv = {(float *)in.data(), length};
    // Prepare output
    auto ret = py::array_t<float>(params.n_mels);
    EmlVector outv = { (float *)ret.data(), params.n_mels };
    // Run
    const EmlError error = eml_audio_melspec(params, inv, outv);
    if (error != EmlOk) {
        throw std::runtime_error("melspec() returned error: " + std::string(eml_error_str(error)));
    }
    return ret;
}


py::array_t<float>
melspectrogram_py(py::array_t<float, py::array::c_style | py::array::forcecast> in,
    int samplerate, int n_fft, int n_mels, float fmin, float fmax
)

{
    if (in.ndim() != 1) {
        throw std::runtime_error("spectrogram input must have dimensions 1");
    }

    if (in.shape(0) != n_fft) {
        throw std::runtime_error("melspectrogram framing not implented");
    }

    // FFT table
    const int n_fft_table = n_fft/2;
    std::vector<float> fft_sin(n_fft_table);
    std::vector<float> fft_cos(n_fft_table);
    EmlFFT fft = { n_fft_table, fft_sin.data(), fft_cos.data() };
    eml_fft_fill(fft, n_fft);

    const EmlAudioMel params = { n_mels, fmin, fmax, n_fft, samplerate };

    // Copy input to avoid modifying
    const int length = (int)in.shape(0);
    std::vector<float> inout(length);
    std::vector<float> temp(length);
    EmlVector inoutv = { (float *)inout.data(), length };
    EmlVector tempv = { (float *)temp.data(), length };
    EmlVector inv = {(float *)in.data(), length};
    eml_vector_set(inoutv, inv, 0);
    eml_vector_set_value(tempv, 0);

    const EmlError error = eml_audio_melspectrogram(params, fft, inoutv, tempv);

    if (error != EmlOk) {
        throw std::runtime_error("melspectrogram returned error: " + std::string(eml_error_str(error)));
    }

    auto ret = py::array_t<float>(params.n_mels);
    EmlVector out = { (float *)ret.data(), params.n_mels };
    eml_vector_set(out, eml_vector_view(inoutv, 0, params.n_mels), 0);

    return ret;
}


py::array_t<float>
sparse_filterbank_py(py::array_t<float, py::array::c_style | py::array::forcecast> in,
    py::array_t<int, py::array::c_style | py::array::forcecast> starts,
    py::array_t<int, py::array::c_style | py::array::forcecast> stops,
    py::array_t<float, py::array::c_style | py::array::forcecast> coeffs
)
{
    EMLPY_PRECONDITION(in.ndim() == 1, "Input must have dim 1");
    EMLPY_PRECONDITION(starts.ndim() == 1, "Starts must have dim 1");
    EMLPY_PRECONDITION(stops.ndim() == 1, "Stops must have dim 1");
    EMLPY_PRECONDITION(coeffs.ndim() == 1, "Coefficients must have dim 1");
    EMLPY_PRECONDITION(starts.shape(0) == stops.shape(0), "Number of starts must equals stops");

    const int output_length = (int)starts.shape(0);
    auto ret = py::array_t<float>(output_length);

    EMLPY_CHECK_ERROR(eml_sparse_filterbank(in.data(),
                        (float *)ret.data(), output_length,
                        starts.data(), stops.data(), coeffs.data()));
    return ret;
}


PYBIND11_MODULE(eml_audio, m) {
    m.doc() = "Audio machine learning for microcontrollers and embedded devices";

    m.def("rfft", rfft_py);
    m.def("melfilter", melfilter_py);

    m.def("melspectrogram", melspectrogram_py);
    m.def("sparse_filterbank", sparse_filterbank_py);

}
