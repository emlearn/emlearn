
#define EML_LOG_ENABLE 1
#define EML_DEBUG

#include <eml_log.h>
#include <eml_audio_features.h>
#include <stdio.h>
#include "./eml_wave.h"

# if 0
typedef struct CheckWindowerData_ {
    int calls;
    int samples;

    int expect_width;
    int expect_length;
    int expect_hop;
} CheckWindowerData;

void
check_windower_callback(EmlArray *arr, void *user_data)
{
    TestWindowerData *data = (TestWindowerData *);

    data->calls += 0;

    int length = arr->dims[0];
    TEST_ASSERT_EQUALS(diff, data->expect_length);

    int16_t *values = (int16_t *)arr->data;

    data->samples += arr->dims[1];

    const int first_value = values[0];
    const int last_value = values[length-1];
    const int diff = last_value - first_value;

    TEST_ASSERT_EQUALS(diff, length);
}

EmlError
fill_array_int16(EmlArray *arr, int start)
{
    EML_PRECONDITION(arr->n_dims == 2, EmlSizeMismatch);

    const int rows = arr->dims[0];
    const int features = arr->dims[1];

    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            int16_t *val = (int16_t *)eml_array_data_2d(arr, i, 0);
            *val = (rows + start);
    }
}

void
check_signal_windower(int window_length, int hop_length, int input_chunk)
{
    const int n_features = 3;

    CheckWindowerData data = { 0 };
    data.expect_length = window_length;
    data.expect_hop = hop_length;
    data.expect_width = n_features;

    EmlSignalWindower _windower;
    EmlSignalWindower *windower = &_windower;


    EmlArray _buffer;
    EmlArray *buffer = &_buffer;
    EML_ARRAY_INIT_2D()

    EmlArray _input;
    EmlArray *input = &_input;
    EML_ARRAY_INIT_2D(input);

    /* FIXME: fill each line of features with different data */
    fill_array_int16(input, 0);
    
    const EmlError init_err = eml_signal_windower_init(windower, buffer);
    TEST_ASSERT_EQUAL(init_err, EmlOk); 

    eml_signal_windower_set_callback(windower, signal_test_windower_callback, &data);

    // TODO: test different hop lengths
    const int n_samples_input = 1;
    for (int i=0; i<samples; i+n_samples_input) {
        // create view
        EML_ARRAY_INIT_2D
        const EmlError add_err = eml_signal_windower_add(windower, chunk);
        TEST_ASSERT_EQUAL(add_err, EmlOk);
    }

    TEST_ASSERT_EQUAL(data.samples, );
    TEST_ASSERT_EQUAL(data.calls, );
}


void
test_signal_windower_hops()
{
    /* window < input chunk */
    // hop=1, input=1
    check_signal_windower(10, 1, 1);

    // hop=1, input=N
    check_signal_windower(10, 1, 5);

    // hop=N, input=1
    check_signal_windower(10, 5, 1);

    // hop=N, input=M
    check_signal_windower(30, 7, 3);

    /* window > input chunk */
    check_signal_windower(10, 1, 15);

    check_signal_windower(10, 5, 18);
}

#endif




typedef struct AudioFeatureExtractor_ {
    
    int samplerate;
    
    EmlSignalWindower windower;
    EmlArray windower_buffer;

    FILE *out_file;
    int64_t sample_counter;

} AudioFeatureExtractor;


void
process_window(EmlArray *arr, void *user_data)
{
    AudioFeatureExtractor *self = (AudioFeatureExtractor *)user_data;
    int16_t *samples = (int16_t *)arr->data;
    const int length = arr->dims[1];
    // FIXME: check other dimension being 1

    /* Compute features */
    const float zcr = eml_signal_zcr_int16(samples, length);
    const float rms = eml_signal_rms_int16(samples, length);
    const float rms_db = eml_signal_power_to_db(rms);
    const float time = self->sample_counter / (float)self->samplerate;

    // Write to output
    fprintf(self->out_file, "%.4f,%.4f,%.4f \n", time, rms_db, zcr);

    EML_LOG_BEGIN("audio-features-process-window");
    EML_LOG_ADD_FLOAT("time", time);
    EML_LOG_ADD_FLOAT("rms_db", rms_db);
    EML_LOG_ADD_FLOAT("zcr", zcr);
    EML_LOG_ADD_INTEGER("sample", self->sample_counter);
    EML_LOG_END();

    /* TODO: add support for exponential averaging / first-order smoothing.
    Use to compute fast time-weighting for soundlevels */

    /* TODO: use EmlIIR to compute A-weighted levels */
    /* TODO: use EmlIIR to extract levels only in speech band */
}

bool
process_samples(int16_t *samples, int length, void *user_data)
{
    AudioFeatureExtractor *self = (AudioFeatureExtractor *)user_data;
    EmlSignalWindower *windower = &self->windower;
    EmlArray chunk;
    EML_ARRAY_INIT_2D(&chunk, 1, length, samples);

    self->sample_counter += length;

    const EmlError windower_add_err = eml_signal_windower_add(windower, &chunk);
    const bool ok = (windower_add_err == EmlOk);
 
#if 0
    EML_LOG_BEGIN("audio-features-process-samples");
    EML_LOG_ADD_INTEGER("length", length);
    EML_LOG_ADD_INTEGER("windower_add_error", windower_add_err);
    EML_LOG_ADD_BOOL("ok", ok);
    EML_LOG_END();
#endif

    return ok;
}

EmlError
audio_feature_extractor_init(AudioFeatureExtractor *self,
        uint8_t *buffer, size_t buffer_length,
        int window_length, int hop_length)
{
    EmlSignalWindower *windower = &self->windower;

    EmlArray *buffer_arr = &self->windower_buffer;
    //eml_array_init_full(
    EML_ARRAY_INIT_2D(buffer_arr, 1, window_length, buffer);

    // FIXME: add a backing buffer

    const EmlError init_err = eml_signal_windower_init(windower, buffer_arr);
    EML_CHECK_ERROR(init_err);

    eml_signal_windower_set_callback(windower, process_window, self);

    windower->window_length = window_length;
    windower->hop_length = hop_length;  

    self->out_file = NULL;
    self->samplerate = 0;

    return EmlOk;
}

EmlError
audio_feature_extractor_process_file(AudioFeatureExtractor *self,
        const char *input, const char *output, int chunk_size)
{
    EML_PRECONDITION(self->samplerate > 0, EmlUninitialized);

    EML_LOG_BEGIN("audio-features-process-file-start");
    //EML_LOG_ADD_INTEGER("args", argc);
    EML_LOG_ADD("input", input);
    EML_LOG_ADD("output", output);
    EML_LOG_END();

    //EmlArray _input;
    //EmlArray *input = &_input;
    //EML_ARRAY_INIT_2D(input);

    // Open .csv output file
    self->out_file = fopen(output, "w");
    if (!self->out_file) {
        return EmlUnknownError;
    }

    // Write header
    fprintf(self->out_file, "%s,%s,%s \n", "time", "rms_db", "zcr");

    const int samplerate = self->samplerate;

    // Open and read .wav file
    // Audio is processed via the callback
    const bool success = \
        eml_wave_read(input, samplerate, chunk_size, process_samples, self);

    if (!success) {
        return EmlUnknownError;
    }

    return EmlOk;
}


#define BUFFER_LENGTH 16000
uint8_t buffer[BUFFER_LENGTH];


// gcc -o test_audio_features test/test_audio_features.c -I./emlearn -lm -lsndfile
int
main(int argc, char *argv[])
{
    AudioFeatureExtractor extractor;

    if (argc != 3) {
        fprintf(stderr, "ERROR: Incorrect number of arguments. Expected 2 \n");
        return -1;
    }

    const char *input = argv[1];
    const char *output = argv[2];


    const int buffer_length = BUFFER_LENGTH;

    // Configuration
    const int samplerate = 16000;
    const int window_length = 16*100;
    const int hop_length = 16*100;
    const int file_chunk_size = 160;

    const EmlError init_err = audio_feature_extractor_init(&extractor,\
        buffer, buffer_length,\
        window_length, hop_length);

    if (init_err != EmlOk) {
        return -3;
    }

    extractor.samplerate = samplerate;

    //test_signal_windower_single_hop();

    EML_LOG_BEGIN("audio-features-main-start");
    EML_LOG_ADD_INTEGER("args", argc);
    EML_LOG_ADD_INTEGER("samplerate", extractor.samplerate);
    EML_LOG_ADD_INTEGER("window_length", window_length);
    EML_LOG_ADD_INTEGER("hop_length", hop_length);
    EML_LOG_END();


    const EmlError process_err = audio_feature_extractor_process_file(&extractor, input, output, file_chunk_size);

    if (process_err != EmlOk) {
        fprintf(stderr, "ERROR: File processing failed \n");
        return -2;
    }

    return 0;
}

