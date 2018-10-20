
#include <stdio.h>

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#include <eml_fft.h>
#include <eml_audio.h>

namespace py = pybind11;


py::array_t<float>
rfft_py(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
    if (in.ndim() != 1) {
        throw std::runtime_error("SFT input must have dimensions 1");
    }

    if (in.shape(0) != EML_AUDIOFFT_LENGTH) {
        throw std::runtime_error("SFT must have length EML_AUDIOFFT_LENGTH");
    }

    auto ret = py::array_t<float>(in.shape(0));

    float *samples = (float *)in.data();
    float *retdata = (float *)ret.data();

    float imag_data[EML_AUDIOFFT_LENGTH];
    EmlVector real = { samples, EML_AUDIOFFT_LENGTH };
    EmlVector imag = { imag_data, EML_AUDIOFFT_LENGTH };
    eml_vector_set_value(imag, 0.0f);

    const int status = eml_audio_fft(real, imag);
 
    eml_vector_set((EmlVector){retdata,EML_AUDIOFFT_LENGTH}, real, 0);

    if (status != 0) {
        throw std::runtime_error("SFT returned error");
    }

    return ret;
}


py::array_t<float>
melspectrogram_py(py::array_t<float, py::array::c_style | py::array::forcecast> in,
    int n_mels, float fmin, float fmax, int n_fft, int samplerate
)

{
    if (in.ndim() != 1) {
        throw std::runtime_error("spectrogram input must have dimensions 1");
    }

    const EmlAudioMel params = { n_mels, fmin, fmax, n_fft, samplerate };

    // Copy input to avoid modifying
    const int length = in.shape(0);
    float inout_data[length];
    float temp_data[length];
    EmlVector inout = { inout_data, length };
    EmlVector temp = { temp_data, length };
    eml_vector_set(inout, (EmlVector){(float *)in.data(), length}, 0);

    const int status = eml_audio_melspectrogram(params, inout, temp);

    if (status != 0) {
        throw std::runtime_error("melspectrogram returned error: " + std::to_string(status));
    }

    auto ret = py::array_t<float>(params.n_mels);
    EmlVector out = { (float *)ret.data(), params.n_mels };
    eml_vector_set(out, eml_vector_view(inout, 0, params.n_mels), 0);

    return ret;
}

PYBIND11_MODULE(eml_audio, m) {
    m.doc() = "Audio machine learning for microcontrollers and embedded devices";

    m.def("rfft", rfft_py);
    m.def("melspectrogram", melspectrogram_py);
}

