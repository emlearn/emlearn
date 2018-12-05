
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
 
    EmlVector retv = { retdata, EML_AUDIOFFT_LENGTH };
    eml_vector_set(retv, real, 0);

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
    std::vector<float> inout(length);
    std::vector<float> temp(length);
    EmlVector inoutv = { (float *)inout.data(), length };
    EmlVector tempv = { (float *)temp.data(), length };
    EmlVector inv = {(float *)in.data(), length};
    eml_vector_set(inoutv, inv, 0);

    const int status = eml_audio_melspectrogram(params, inoutv, tempv);

    if (status != 0) {
        throw std::runtime_error("melspectrogram returned error: " + std::to_string(status));
    }

    auto ret = py::array_t<float>(params.n_mels);
    EmlVector out = { (float *)ret.data(), params.n_mels };
    eml_vector_set(out, eml_vector_view(inoutv, 0, params.n_mels), 0);

    return ret;
}

class Processor {
private:
    BirdDetector detector;
    EmlAudioBufferer bufferer;

    float record1[AUDIO_HOP_LENGTH];
    float record2[AUDIO_HOP_LENGTH];
    float input_data[AUDIO_WINDOW_LENGTH];
    float temp1_data[AUDIO_WINDOW_LENGTH];
    float temp2_data[AUDIO_WINDOW_LENGTH];

public:
    Processor(int n_mels, float fmin, float fmax, int n_fft, int samplerate) {

        bufferer = (EmlAudioBufferer){ AUDIO_HOP_LENGTH, record1, record2, NULL, NULL, 0 };

        const EmlAudioMel params = {
            n_mels: n_mels,
            fmin: fmin,
            fmax: fmax,
            n_fft:AUDIO_WINDOW_LENGTH,
            samplerate:samplerate,
        };

        const int features_length = params.n_mels; // only 1 feature per mel band right now
        float features_data[features_length];

        detector = {
            audio: (EmlVector){ input_data, AUDIO_WINDOW_LENGTH },
            features: (EmlVector){ features_data, features_length },
            temp1: (EmlVector){ temp1_data, AUDIO_WINDOW_LENGTH },
            temp2: (EmlVector){ temp2_data, AUDIO_WINDOW_LENGTH },
            mel_filter: params,
            model: birddetect_model,
        };

        reset();
    }

    void reset() {
        birddetector_reset(&detector);
        eml_audio_bufferer_reset(&bufferer);
    }

    int add_samples(py::array_t<float, py::array::c_style | py::array::forcecast> in) {
        if (in.ndim() != 1) {
            throw std::runtime_error("process() input must have dimensions 1");
        }

        float *samples = (float *)in.data();
        int length = in.shape(0);

        int n_frames = 0;
        for (int i=0; i<length; i++) {
            eml_audio_bufferer_add(&bufferer, samples[i]);

            if (bufferer.read_buffer) {
                EmlVector frame = { bufferer.read_buffer, bufferer.buffer_length };
                birddetector_push_frame(&detector, frame);
                bufferer.read_buffer = NULL; // done processing

                n_frames += 1;
            }

        }
        return n_frames;
    }

    py::array_t<float> get_features() {

        const int length = detector.features.length;
        auto ret = py::array_t<float>(length);
        EmlVector out = { (float *)ret.data(), length };
        eml_vector_set(out, detector.features, 0);

        return ret;
    }

    bool predict() {
        return birddetector_predict(&detector);
    }

};


PYBIND11_MODULE(eml_audio, m) {
    m.doc() = "Audio machine learning for microcontrollers and embedded devices";

    m.def("rfft", rfft_py);
    m.def("melspectrogram", melspectrogram_py);

    py::class_<Processor>(m, "Processor")
        .def(py::init< int, float, float, int, int >())
        .def("add_samples", &Processor::add_samples)
        .def("predict", &Processor::predict)
        .def("get_features", &Processor::get_features)
        .def("reset", &Processor::reset);
}

