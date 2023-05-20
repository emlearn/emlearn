
#ifndef EML_AUDIO_H
#define EML_AUDIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "eml_common.h"
#include "eml_vector.h"
#include "eml_fft.h" 

#include <math.h>


// Power spectrogram
// TODO: operate in-place
EmlError
eml_audio_power_spectrogram(EmlVector rfft, EmlVector out, int n_fft) {
    const int spec_length = 1+n_fft/2;

    EML_PRECONDITION(rfft.length > spec_length, EmlSizeMismatch);
    EML_PRECONDITION(out.length == spec_length, EmlSizeMismatch);

    const float scale = 1.0f/n_fft;
    for (int i=0; i<spec_length; i++) {
        const float a = (float)fabs(rfft.data[i]);
        out.data[i] = scale * powf(a, 2);
    }
    return EmlOk;
}

// Simple formula, from Hidden Markov Toolkit
// in librosa have to use htk=True to match
float
eml_audio_mels_from_hz(float hz) {
    return (float)(2595.0 * log10(1.0 + (hz / 700.0)));
}
float
eml_audio_mels_to_hz(float mels) {
    return (float)(700.0 * (powf(10.0, (float)(mels/2595.0)) - 1.0));
}


typedef struct _EmlAudioMel {
    int n_mels;
    float fmin;
    float fmax;
    int n_fft;
    int samplerate;
} EmlAudioMel;


float
eml_audio_mel_center(EmlAudioMel params, int n) {
    // Filters are spaced evenly in mel space
    const float melmin = eml_audio_mels_from_hz(params.fmin);
    const float melmax = eml_audio_mels_from_hz(params.fmax);
    const float melstep = (melmax-melmin)/(params.n_mels+1);

    const float mel = melmin + (n * melstep);
    const float hz = eml_audio_mels_to_hz(mel);
    return hz;
}
int
eml_audio_mel_bin(EmlAudioMel params, float hz) {
    const int bin = (int)floor((params.n_fft+1)*(hz/params.samplerate));
    return bin;
}
static int
mel_bin(EmlAudioMel params, int n) {
    const float hz = eml_audio_mel_center(params, n);
    return eml_audio_mel_bin(params, hz);
}

float
eml_fft_freq(EmlAudioMel params, int n) {
    const float end = params.samplerate/2.0f;
    const int steps = (1+params.n_fft/2) - 1;
    return (n*end)/steps;
}


EmlError
eml_audio_melspec(EmlAudioMel mel, EmlVector spec, EmlVector mels) {

    const int max_bin = 1+mel.n_fft/2;
    EML_PRECONDITION(max_bin <= spec.length, EmlSizeMismatch);
    EML_PRECONDITION(mel.n_mels == mels.length, EmlSizeMismatch);

    // Note: no normalization
    for (int m=1; m<mel.n_mels+1; m++) {
        const int left = mel_bin(mel, m-1);
        const int center = mel_bin(mel, m);
        const int right = mel_bin(mel, m+1);

        if (left < 0) {
            return EmlUnknownError;
        }
        if (right > max_bin) {
            return EmlUnknownError;
        }

        const float fdifflow = eml_audio_mel_center(mel, m) - eml_audio_mel_center(mel, m-1);
        const float fdiffupper = eml_audio_mel_center(mel, m+1) - eml_audio_mel_center(mel, m);

        //fprintf(stderr, "mel %d:(%d, %d, %d) \n", m, left, center, right);

        float val = 0.0f;
        for (int k=left; k<=center; k++) {
            const float r = eml_audio_mel_center(mel, m-1) - eml_fft_freq(mel, k);
            const float weight = eml_max(eml_min(-r/fdifflow, 1.0f), 0.0f);
            //if (m == 2) {
            //    fprintf(stderr, "k=%d wl=%f \n", k, weight);
            //}
            val += spec.data[k] * weight;
        }
        for (int k=center; k<right; k++) {
            const float r = eml_audio_mel_center(mel, m+1) - eml_fft_freq(mel, k+1);
            const float weight = eml_max(eml_min(r/fdiffupper, 1.0f), 0.0f);
            //if (m == 2) {
            //    fprintf(stderr, "k=%d, wr=%f \n", k, weight);
            //}
            val += spec.data[k+1] * weight;
        }
        //fprintf(stderr, "mel %d: val=%f\n", m, val);

        mels.data[m-1] = val;
    }

    return EmlOk;
}


EmlError
eml_audio_melspectrogram(EmlAudioMel mel_params, EmlFFT fft, EmlVector inout, EmlVector temp)
{
    const int n_fft = mel_params.n_fft;
    const int s_length = 1+n_fft/2;
    const int n_mels = mel_params.n_mels;
 
    // Apply window
    EML_CHECK_ERROR(eml_vector_hann_apply(inout));

    // Perform (short-time) FFT
    EML_CHECK_ERROR(eml_vector_set_value(temp, 0.0f));
    EML_CHECK_ERROR(eml_fft_forward(fft, inout.data, temp.data, inout.length));

    // Compute mel-spectrogram
    EML_CHECK_ERROR(eml_audio_power_spectrogram(inout, eml_vector_view(temp, 0, s_length), n_fft));
    EML_CHECK_ERROR(eml_audio_melspec(mel_params, temp, eml_vector_view(inout, 0, n_mels)));

    return EmlOk;
}

/*
Apply a sparse filterbank which reduces @input to a smaller @output

Each filter is on form 0000nonzero0000
The nonzero filter coefficients are stored consecutively in @lut,
with @start and @end indicating which index (in the input) each filter start/end at

Typically the filters are triangular and applied to an FFT power spectrum
Can be used for mel-filtering a spectrogram
*/
EmlError
eml_sparse_filterbank(const float *input,
             float *output, int output_length,
             const int *starts, const int *stops, const float *lut)
{
    for (int i=0; i<output_length; i++) {
        output[i] = 0.0f;
    }

    int offset = 0;
    for (int i = 0; i < output_length; i++) {
        const int start = starts[i];
        const int stop = stops[i];

        //EML_PRECONDITION(start > 0, EmlUninitialized);
        //EML_PRECONDITION(stop > 0, EmlUninitialized);

        for (int j = start; j <= stop; j++) {
            const float f = lut[offset];
            //EML_PRECONDITION(f > 0, EmlUninitialized);
            output[i] += input[j] * f;
            offset++;
        }
    }

    return EmlOk;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_AUDIO_H
