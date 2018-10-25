
#include <stdio.h>

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#define EML_DEBUG 1

#include <eml_dsp.h>
#include <eml_fft.h>
#include <eml_audio.h>

namespace py = pybind11;


py::array_t<float>
rfft_py(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
    if (in.ndim() != 1) {
        throw std::runtime_error("SFT input must have dimensions 1");
    }

    const int n_fft = in.shape(0);

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
    const int length = in.shape(0);
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
    const int length = in.shape(0);
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


// Returns an EmlNetActivationFunction or -EmlError
int32_t
eml_convolve_mode(const char *str)
{
    int32_t ret = -EmlUnsupported;

    for (int i=0; i<EmlConvolveModes; i++) {
        const char *func_str = eml_convolve_mode_strs[i];
        if (strcmp(str, func_str) == 0) {
            ret = (int32_t)i;
        }
    }

    return ret;
}

void
assert_no_error(int32_t code, std::string err_str) {
    const EmlError err = (EmlError)code;
    if (err > EmlOk) {
        throw std::runtime_error(err_str);
    }
}
void
assert_no_error(int32_t code) {
    const EmlError err = (EmlError)code;
    if (err > EmlOk) {
        std::string err_str = eml_error_str(err);
        throw std::runtime_error("emlearn error: " + err_str);
    }
}

py::array_t<float>
convolve1d(py::array_t<float, py::array::c_style | py::array::forcecast> in,
           py::array_t<float, py::array::c_style | py::array::forcecast> kernel,
           std::string mode_name, int32_t stride)
{
    if (in.ndim() != 1) {
        throw std::runtime_error("input must have dimensions 1");
    }
    if (kernel.ndim() != 1) {
        throw std::runtime_error("kernel must have dimensions 1");
    }
    const int32_t mode_int = eml_convolve_mode(mode_name.c_str());
    assert_no_error(-mode_int, "unsupported convolve mode: " + mode_name);
    const EmlConvolveMode mode = (EmlConvolveMode)mode_int;

    const int32_t in_length = in.shape(0);
    const int32_t kernel_size = kernel.shape(0);
    const int32_t ret_length = eml_convolve_1d_length(in_length, kernel_size, mode);
    auto ret = py::array_t<float>(ret_length);

    fprintf(stderr, "mode=%d, ret_length=%d\n", mode_int, ret_length);

    const EmlError status = eml_convolve_1d((float *)in.data(), in_length,
                                (float *)ret.data(), ret_length,
                                (float *)kernel.data(), kernel_size,
                                stride, mode);
    assert_no_error(status);

    return ret;
}


PYBIND11_MODULE(eml_audio, m) {
    m.doc() = "Audio machine learning for microcontrollers and embedded devices";

    m.def("rfft", rfft_py);

    m.def("melfilter", melfilter_py);
    m.def("melspectrogram", melspectrogram_py);


    m.def("convolve1d", convolve1d);
}

