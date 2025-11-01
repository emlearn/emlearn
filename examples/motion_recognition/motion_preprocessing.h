
#include <math.h>
#include <eml_iir.h>
#include <eml_fft.h>

#define ACCELGYRO_INPUT_CHANNELS 6

// IIR filters for gravity estimation are biquads, so 2 stages -> 4th order
#define ACCELGYRO_GRAVITY_FILTER_STAGES 2
#define ACCELGYRO_GRAVITY_FILTER_COEFFICIENTS (ACCELGYRO_GRAVITY_FILTER_STAGES*6)
#define ACCELGYRO_GRAVITY_FILTER_STATES (ACCELGYRO_GRAVITY_FILTER_STAGES*4)

#ifndef ACCELGYRO_FFT_LENGTH
#define ACCELGYRO_FFT_LENGTH 64
#endif
#define ACCELGYRO_FFT_TABLE_LENGTH (ACCELGYRO_FFT_LENGTH/2)


enum accelgyro_feature {
    accelgyro_feature_orientation_x = 0,
    accelgyro_feature_orientation_y,
    accelgyro_feature_orientation_z,
    accelgyro_feature_motion_mag_rms,
    accelgyro_feature_motion_mag_p2p,
    accelgyro_feature_motion_x_rms,
    accelgyro_feature_motion_y_rms,
    accelgyro_feature_motion_z_rms,
    accelgyro_features_length
};
//#define ACCELGYRO_PREPROCESSOR_FEATURES 6

const char *accelgyro_feature_names[accelgyro_features_length+1] = {
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "motion_mag_rms",
    "motion_mag_p2p",
    "motion_x_rms",
    "motion_y_rms",
    "motion_z_rms",
    "LENGTH_NOT_FEATURE"
};

struct accelgyro_preprocessor {

    int samplerate;
    int window_length;

    // output buffer
    float features[accelgyro_features_length];

    // decomposed gravity and motion (linear acceleration) vectors
    float motion[3];
    float gravity[3];

    int32_t frames_processed;

    // IIR filter for gravity separation
    // Multiple filters and associated states in XYZ order
    // Coefficients are shared, same for X,Y,Z
    EmlIIR gravity_filters[3];
    float gravity_coefficients[ACCELGYRO_GRAVITY_FILTER_COEFFICIENTS];
    float gravity_states[3*ACCELGYRO_GRAVITY_FILTER_STATES];
    bool gravity_filter_enable;

    // FFT for frequency-domain feature extraction
    EmlFFT fft;
    int fft_length;
    // the range of FFT bins to output as features
    int fft_feature_start;
    int fft_feature_end;
    // pre-computed table of coefficients
    float fft_sin[ACCELGYRO_FFT_TABLE_LENGTH];
    float fft_cos[ACCELGYRO_FFT_TABLE_LENGTH];
    // FFT data buffers
    float fft_real[ACCELGYRO_FFT_LENGTH];
    float fft_imag[ACCELGYRO_FFT_LENGTH];
};

int
accelgyro_preprocessor_init(struct accelgyro_preprocessor *self, int samplerate, int window_length)
{
    const int fft_length = ACCELGYRO_FFT_LENGTH;
    if (fft_length != 0 && window_length >= fft_length) {
        return -1;
    }
    self->fft_length = fft_length;

    self->samplerate = samplerate;
    self->window_length = window_length;

    self->frames_processed = 0;

    // Gravity separation
    self->gravity_filter_enable = false;

    for (int i=0; i<3; i++) {
        self->gravity_filters[i] = (EmlIIR){
            ACCELGYRO_GRAVITY_FILTER_STAGES,
            self->gravity_states + (i * ACCELGYRO_GRAVITY_FILTER_STATES),
            ACCELGYRO_GRAVITY_FILTER_STATES,
            self->gravity_coefficients,
            ACCELGYRO_GRAVITY_FILTER_COEFFICIENTS,
        };

        const EmlError filter_err = eml_iir_check(self->gravity_filters[i]);
        if (filter_err != EmlOk) {
            return -2;
        }

    }
    for (int i=0; i<3*ACCELGYRO_GRAVITY_FILTER_STATES; i++) {
        self->gravity_states[i] = 0.0f;
    }

    // FFT
    self->fft = (EmlFFT){ ACCELGYRO_FFT_TABLE_LENGTH, self->fft_sin, self->fft_cos };
    const EmlError fill_err = eml_fft_fill(self->fft, self->fft_length);
    if (fill_err != EmlOk) {
        return -3;
    }
    // Fill with zeros. Especially important since our window_length might be < fft_length
    for (int i=0; i<fft_length; i++) {
        self->fft_real[i] = 0.0f;
        self->fft_imag[i] = 0.0f;
    }
    self->fft_feature_start = 0;
    self->fft_feature_end = self->fft_length;

    return 0;
}

int
accelgyro_preprocessor_get_feature_length(struct accelgyro_preprocessor *self)
{
    const int fixed_features = accelgyro_features_length;
    const int fft_features = self->fft_feature_end - self->fft_feature_start;
    const int total_features = fixed_features + fft_features;

    return total_features;
}

// On success, returns number of characters written (including the \0 byte)
// On failure, returns a negative error code
int
accelgyro_preprocessor_get_feature_name(struct accelgyro_preprocessor *self,
        int index, char *out, size_t length)
{
    const int n_features = accelgyro_preprocessor_get_feature_length(self);
    if (n_features < 0) {
        return -1;
    }
    if (index < 0) {
        return -2;
    }
    if (index > n_features-1) {
        return -2;
    }

    if (index < accelgyro_features_length) {
        // regular fixed feature
        const char * feature_name = accelgyro_feature_names[index];
        const int needed = snprintf(out, length, "%s", feature_name);
        if (needed < 0) {
            // error
            out[0] = '\0';
            return -4;
        } else if (needed >= length) {
            // truncated
            return -5;
        } else {
            // success
            return needed+1;
        }
    } else {
        // FFT feature

        // Find the frequency for the FFT bin in question
        const int fft_index = index - accelgyro_features_length;
        const int fft_bin = fft_index + self->fft_feature_start;
        const float freq = fft_bin * (self->samplerate/(float)self->fft_length);
        float freq_integer;
        float freq_frac = modff(fabsf(freq), &freq_integer);
        int freq_first_decimal = (int)roundf(freq_frac * powf(10.0f, 1));

        const int needed = snprintf(out, length, "fft_%d_%dhz",
            (int)freq_integer, freq_first_decimal);
        if (needed < 0) {
            // error
            out[0] = '\0';
            return -6;
        } else if (needed >= length) {
            // truncated
            return -7;
        } else {
            // success
            return needed+1;
        }
    }

    return -5;
}

int
accelgyro_preprocessor_get_features(struct accelgyro_preprocessor *self, float *out, size_t length)
{
    const int n_features = accelgyro_preprocessor_get_feature_length(self);
    if (n_features < 0) {
        return -2;
    }
    if (length < n_features) {
        return -1;
    }

    // Copy regular features
    memcpy(out, self->features, accelgyro_features_length*sizeof(float));

    // Copy FFT features
    const float *fft_start = self->fft_real + self->fft_feature_start;
    const int fft_items = self->fft_feature_end - self->fft_feature_start;
    memcpy(out+accelgyro_features_length, fft_start, fft_items*sizeof(float));

    return 0;
}

// Configure the lowpass filter used to estimate gravity
// coeff must be for a 4th-order IIR filter, on EmlIIR format
int
accelgyro_preprocessor_set_gravity_lowpass(struct accelgyro_preprocessor *self,
        const float *coeff, int n_coefficients)
{
    if (n_coefficients != ACCELGYRO_GRAVITY_FILTER_COEFFICIENTS) {
        return -1;
    }

    memcpy(self->gravity_coefficients, coeff, n_coefficients*sizeof(float));
    self->gravity_filter_enable = true;

    return 0;
}

// Configure which FFT bins should be included as features. Range: [start, end-1]
int
accelgyro_preprocessor_set_fft_features(struct accelgyro_preprocessor *self,
        int start, int end)
{
    if (start < 0 || end < 0) {
        return -1;
    }
    if (self->fft_length != 0 && end > self->fft_length) {
        return -2;
    }
    if (self->fft_length != 0 && start > self->fft_length) {
        return -3;
    }
    
    self->fft_feature_start = start;
    self->fft_feature_end = end;

    return 0;
}


int
accelgyro_preprocessor_run(struct accelgyro_preprocessor *self,
                            const float *data,
                            int length)
{
    const int expect_length = self->window_length * ACCELGYRO_INPUT_CHANNELS;
    if (length != expect_length) {
        return -1;
    }

    if (self->samplerate <= 0) {
        // Invalid samplerate. Not initialized?
        return -1;
    }

    float motion_x_squared = 0.0;
    float motion_y_squared = 0.0;
    float motion_z_squared = 0.0;

    float motion_mag_squared = 0.0f;
    float motion_mag_min = INFINITY;
    float motion_mag_max = -INFINITY;

    if (self->frames_processed <= 0) {
        // handle overflow
        self->frames_processed = 0;
    }

    // Clear FFT
    for (int i=0; i<self->fft_length; i++) {
        self->fft_real[i] = 0.0f;
        self->fft_imag[i] = 0.0f;
    }

    const int n_frames = length / ACCELGYRO_INPUT_CHANNELS;
    for (int frame=0; frame<n_frames; frame++) {

        // NOTE: accelerometer XYZ must be first 3 components
        const int offset = frame * ACCELGYRO_INPUT_CHANNELS;
        const float *xyz = data+offset;

        // NOTE: gyro data currently ignored
        // TODO: do sensor-fusion with gyro, and use complimentary filter for gravity separation
    
        if (self->gravity_filter_enable) {

            // Gravity vector separation using low-pass
            if (self->frames_processed == 0) {
                // warm up the low-pass filter
                // avoids slow ramp-in from startup 0 to the near-DC values
                const int initialization_repetitions = 10;
                for (int i=0; i<3; i++) {
                    for (int r=0; r<initialization_repetitions; r++) {
                        eml_iir_filter(self->gravity_filters[i], xyz[i]);
                    }
                }
            }

            // Estimate gravity with low-pass,
            // and subtract gravity to estimate linear acceleration / "motion"
            for (int i=0; i<3; i++) {
                self->gravity[i] = eml_iir_filter(self->gravity_filters[i], xyz[i]);
                self->motion[i] = xyz[i] - self->gravity[i];
            }
        } else {
            // No gravity filter. XXX: motion will be 0 in this case
            for (int i=0; i<3; i++) {
                self->gravity[i] = xyz[i];
                self->motion[i] = xyz[i] - self->gravity[i];
            }

        }

        const float motion_x = self->motion[0];
        const float motion_y = self->motion[1];
        const float motion_z = self->motion[2];

        // magnitude squared for rms
        const float motion_mag = sqrtf((motion_x*motion_x) + (motion_y*motion_y) + (motion_z*motion_z));
        motion_mag_squared += (motion_mag*motion_mag);

        // motion min/max for p2p
        if (motion_mag < motion_mag_min) {
            motion_mag_min = motion_mag;
        }
        if (motion_mag > motion_mag_max) {
            motion_mag_max = motion_mag;
        }

        // XYZ motion
        motion_x_squared += (motion_x*motion_x);
        motion_y_squared += (motion_y*motion_y);
        motion_z_squared += (motion_z*motion_z);

        // Prepare for FFT
        if (self->fft_length != 0) {
            self->fft_real[frame] = motion_mag;
            self->fft_imag[frame] = 0.0f;
        }

        // Internal metric tracking
        self->frames_processed += 1;
    }

    float *features = self->features;

    // Orientation vector
    // Is the gravity vector estimate, normalized to 1.0 magitude
    const float *gv = self->gravity;
    const float gravity_mag = sqrtf((gv[0]*gv[0]) + (gv[1]*gv[1]) + (gv[2]*gv[2]));
    features[accelgyro_feature_orientation_x] = self->gravity[0] / gravity_mag;
    features[accelgyro_feature_orientation_y] = self->gravity[1] / gravity_mag;
    features[accelgyro_feature_orientation_z] = self->gravity[2] / gravity_mag;

    // Motion in XYZ
    features[accelgyro_feature_motion_x_rms] = sqrtf(motion_x_squared / n_frames);
    features[accelgyro_feature_motion_y_rms] = sqrtf(motion_y_squared / n_frames);
    features[accelgyro_feature_motion_z_rms] = sqrtf(motion_z_squared / n_frames);

    // Motion magnitude RMS
    const float motion_mag_rms = sqrtf(motion_mag_squared / n_frames);
    features[accelgyro_feature_motion_mag_rms] = motion_mag_rms;

    // Motion magnitude p2p
    const float motion_mag_p2p = motion_mag_max - motion_mag_min;
    features[accelgyro_feature_motion_mag_p2p] = motion_mag_p2p;

    // Perform FFT
    if (self->fft_length != 0) {

#if 0
        printf("FFT [");
        for (int i=0; i<self->fft_length; i++) {
            printf("%.2f ", self->fft_real[i]);
        }
        printf("]\n");
#endif

        const EmlError fft_err = \
            eml_fft_forward(self->fft, self->fft_real, self->fft_imag, self->fft_length);
        if (fft_err != EmlOk) {
            return -2;
        }

        // Convert to magnitude
        for (int i=0; i<self->fft_length; i++) {
            self->fft_real[i] = fabs(self->fft_real[i]);
        }

#if 0
        fprintf(stderr, "fft-run nfft=%d window=%d\n",
            self->fft_length, length/ACCELGYRO_INPUT_CHANNELS);
#endif

    }

    return 0;
}

