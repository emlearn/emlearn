
#ifndef EML_AUDIO_H
#define EML_AUDIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "eml_vector.h"

// Double buffering
typedef struct _EmlAudioBufferer {
    int buffer_length;
    float *buffer1;
    float *buffer2;

    float *write_buffer;
    float *read_buffer;
    int write_offset;
} EmlAudioBufferer;

void
eml_audio_bufferer_reset(EmlAudioBufferer *self) {
    self->write_buffer = self->buffer1;
    self->read_buffer = NULL;
    self->write_offset = 0;
}

int
eml_audio_bufferer_add(EmlAudioBufferer *self, float s) {

    self->write_buffer[self->write_offset++] = s; 

    if (self->write_offset == self->buffer_length) {

        if (self->read_buffer) {
            // consumer has not cleared it
            return -1;
        }

        self->write_offset = 0;
        self->read_buffer = self->write_buffer;
        self->write_buffer = (self->read_buffer == self->buffer1) ? self->buffer2 : self->buffer1;
        return 1;
    } else {
        return 0;
    }
}

#define EML_AUDIOFFT_LENGTH 1024
#define FFT_TABLE_SIZE EML_AUDIOFFT_LENGTH/2

int
eml_audio_fft(EmlVector real, EmlVector imag) {  

    if (real.length != EML_AUDIOFFT_LENGTH) {
        return -1;
    }
    if (imag.length != EML_AUDIOFFT_LENGTH) {
        return -2;
    }

    const bool success = eml_fft_transform(real.data, imag.data, EML_AUDIOFFT_LENGTH);
    if (!success) {
        return -3;
    }

    return 0;
}

// Power spectrogram
// TODO: operate in-place
int
eml_audio_power_spectrogram(EmlVector rfft, EmlVector out, int n_fft) {
    const int spec_length = 1+n_fft/2;

    if (rfft.length < spec_length) {
        return -1;
    }
    if (out.length != spec_length) {
        return -2;
    }

    const float scale = 1.0f/n_fft;
    for (int i=0; i<spec_length; i++) {
        const float a = fabs(rfft.data[i]);
        out.data[i] = scale * powf(a, 2);
    }
    return 0;
}

// Simple formula, from Hidden Markov Toolkit
// in librosa have to use htk=True to match
float
eml_audio_mels_from_hz(float hz) {
    return 2595.0 * log10(1.0 + (hz / 700.0));
}
float
eml_audio_mels_to_hz(float mels) {
    return 700.0 * (powf(10.0, mels/2595.0) - 1.0);
}


typedef struct _EmlAudioMel {
    int n_mels;
    float fmin;
    float fmax;
    int n_fft;
    int samplerate;
} EmlAudioMel;


static int
mel_bin(EmlAudioMel params, int n) {

    // Filters are spaced evenly in mel space
    const float melmin = eml_audio_mels_from_hz(params.fmin);
    const float melmax = eml_audio_mels_from_hz(params.fmax);
    const float melstep = (melmax-melmin)/(params.n_mels+1);

    const float mel = melmin + (n * melstep);
    const float hz = eml_audio_mels_to_hz(mel);
    const int bin = floor((params.n_fft+1)*(hz/params.samplerate));
    return bin;
}


// https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
int
eml_audio_melspec(EmlAudioMel mel, EmlVector spec, EmlVector mels) {

    const int max_bin = 1+mel.n_fft/2;
    if (max_bin > spec.length) {
        return -1;
    }
    if (mel.n_mels != mels.length) {
        return -2;
    }

    // Note: no normalization

    for (int m=1; m<mel.n_mels+1; m++) {
        const int left = mel_bin(mel, m-1);
        const int center = mel_bin(mel, m);
        const int right = mel_bin(mel, m+1);
    
        if (left < 0) {
            return -3;
        }
        if (right > max_bin) {
            return -4;
        } 

        float val = 0.0f;
        for (int k=left; k<center; k++) {
            const float weight = (float)(k - left)/(center - left);
            val += spec.data[k] * weight;
        }
        for (int k=center; k<right; k++) {
            const float weight = (float)(right - k)/(right - center);
            val += spec.data[k] * weight;
        }

        mels.data[m-1] = val;
    }

    return 0;
}



#define EM_RETURN_IF_ERROR(expr) \
    do { \
        const int _e = (expr); \
        if (_e != 0) { \
            return _e; \
        } \
    } while(0);

int
eml_audio_melspectrogram(EmlAudioMel mel_params, EmlVector inout, EmlVector temp) {

    const int n_fft = mel_params.n_fft;
    const int s_length = 1+n_fft/2;
    const int n_mels = mel_params.n_mels;
 
    // Apply window
    EM_RETURN_IF_ERROR(eml_vector_hann_apply(inout));

    // Perform (short-time) FFT
    EM_RETURN_IF_ERROR(eml_vector_set_value(temp, 0.0f));
    EM_RETURN_IF_ERROR(eml_audio_fft(inout, temp));

    // Compute mel-spectrogram
    EM_RETURN_IF_ERROR(eml_audio_power_spectrogram(inout, eml_vector_view(temp, 0, s_length), n_fft));
    EM_RETURN_IF_ERROR(eml_audio_melspec(mel_params, temp, eml_vector_view(inout, 0, n_mels)));

    return 0;
}


// birddetector.h
typedef struct _BirdDetector {
    EmlVector audio;
    EmlVector features;
    EmlVector temp1;
    EmlVector temp2;
    EmlAudioMel mel_filter;
    EmlTreesModel *model;
} BirdDetector;


void
birddetector_reset(BirdDetector *self) {

    eml_vector_set_value(self->audio, 0.0f);
    eml_vector_set_value(self->features, 0.0f);
    eml_vector_set_value(self->temp1, 0.0f);
    eml_vector_set_value(self->temp2, 0.0f);
}

void
birddetector_push_frame(BirdDetector *self, EmVector frame) {

    // insert new frame into our buffer
    eml_vector_shift(self->audio, -frame.length);
    eml_vector_set(self->audio, frame, self->audio.length-frame.length);

    // process current window
    eml_vector_set(self->temp1, self->audio, 0);
    eml_audio_melspectrogram(self->mel_filter, self->temp1, self->temp2);
    eml_vector_meansub(self->temp1);

    // Feature summarization
    eml_vector_max_into(self->features, emvector_view(self->temp1, 0, self->features.length));
}

bool
birddetector_predict(BirdDetector *self) {

    const int32_t cl = eml_trees_predict(self->model, self->features.data, self->features.length);
    return cl == 1;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_AUDIO_H
