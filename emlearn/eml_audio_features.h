
#include <eml_common.h>
#include <eml_array.h>

#include <math.h>

#define EML_SIGNAL_SIGN(s) ((s < 0) ? 1 : 0)

float
eml_signal_db_to_power(float db) {
    return powf(10.0f, db/10.0f);
}

float
eml_signal_power_to_db(float power) {
    return 10.0f * log10f(power);
}

/*
Returns the RMS (root-mean-squared) power of a signal
*/
float
eml_signal_rms_int16(int16_t *pcm, int length)
{
    float sum;
    for (int i=0; i<length; i++) {
        const float s = pcm[i]/32768.0f;
        sum += powf(s, 2);
    }
    const float mean = sum / length;
    const float rms = sqrtf(mean);
    return rms;
}



/* 
Returns fractions of zero-crossings in the window
*/
float
eml_signal_zcr_int16(int16_t *pcm, int length)
{
    int crossings = 0;

	for (int i=0; i<length; i++) {
		const int sign1 = EML_SIGNAL_SIGN(pcm[i]);
		const int sign2 = EML_SIGNAL_SIGN(pcm[i+1]);
		if (sign1 != sign2) {
            crossings += 1;
        }
	}

    const float rate = crossings/length;
    return rate;
}

/* 
EmlSignalWindower. Process an input stream as overlapped fixed-length windows 

Usage examples:
# microphone input
inputs. PCM samples. int16_t of (16, 1). 1000 times per second for 16kHz audio
outputs. Windows ready for feature extraction. ZCR,RMS etc

# accelerometer input. 
inputs. Accelerometer XYZ samples, 100 hz. 32 samples FIFO, 3 times per seconds. (32, 3)
outputs. Ready for FFT/STFT. (512, 3)

# Spectrogram input
inputs. mel-spectrogram bands, 32 bands. (1, 32)
output. window. Ready for CNN input (30, 32)

*/

typedef void (*EmlSignalWindowerDataCallback)(EmlArray *, void *);

typedef struct EmlSignalWindower_ {

    /* Configuration */
    int window_length;
    int hop_length;
    void *user_data;
    EmlSignalWindowerDataCallback callback;

    /* State */
    EmlArray *buffer;
    int valid_samples;

} EmlSignalWindower; 


EmlError
eml_signal_windower_init(EmlSignalWindower *self, EmlArray *arr) 
{
    EML_PRECONDITION(arr->n_dims == 2, EmlSizeMismatch);
    self->buffer = arr; 

    // config
    self->hop_length = 1;
    self->window_length = 1;

    // state
    self->valid_samples = 0;
    self->callback = NULL;
    self->user_data = NULL;

    return EmlOk;
}

void
eml_signal_windower_set_callback(EmlSignalWindower *self,
            EmlSignalWindowerDataCallback callback, void *user_data)
{
    self->callback = callback;
    self->user_data = user_data;
}


EmlError
eml_signal_windower_add(EmlSignalWindower *self, const EmlArray *arr)
{
    EML_PRECONDITION(arr->n_dims == 2, EmlSizeMismatch);
    const int features = arr->dims[0];
    const int samples = arr->dims[1];
    EML_PRECONDITION(features == self->buffer->dims[0], EmlSizeMismatch);
    EML_PRECONDITION(samples <= self->buffer->dims[1], EmlSizeMismatch);

    // input length might be
    // smaller or larger than hop_length
    // Might not be divisible by hop

#if 0
    EML_LOG_BEGIN("eml_signal_windower_add");
    EML_LOG_ADD_INTEGER("features", features);
    EML_LOG_ADD_INTEGER("samples", samples);
    EML_LOG_ADD_INTEGER("valid_samples", self->valid_samples);
#if 0
    EML_LOG_ADD_INTEGER("window_length", self->window_length);
    EML_LOG_ADD_INTEGER("hop_length", self->hop_length);
#endif
    EML_LOG_END();
#endif

    int current = 0;
    while (samples-current > 0) {

        // Fill inn new data
        const int shift = fmin(self->hop_length, samples-current);
        current += shift;
        eml_array_copy_rows(self->buffer, self->valid_samples, arr);
        self->valid_samples += shift;
#if 0
        EML_LOG_BEGIN("eml_signal_windower_add_iter");
        EML_LOG_ADD_INTEGER("valid_samples", self->valid_samples);
        EML_LOG_ADD_INTEGER("shift", shift);
        EML_LOG_END();
#endif

        if (self->valid_samples >= self->window_length) {

            // A full window is available
            // Create a view on the data
            EmlArray chunk;
            EML_ARRAY_INIT_2D(&chunk, self->window_length, features, self->buffer->data);

            // Call the processing callback
            if (self->callback) {
                self->callback(&chunk, self->user_data);
            }

            // Shift out old data
            self->valid_samples -= self->hop_length;
            eml_array_shift_rows(self->buffer, -self->hop_length);
        }

    }

    return EmlOk;
}



