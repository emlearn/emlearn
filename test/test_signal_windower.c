

#include <eml_array.h>
#include <eml_audio_features.h>

#include <unity.h>


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
    CheckWindowerData *data = (CheckWindowerData *)user_data;

    data->calls += 0;

    const int length = arr->dims[0];
    TEST_ASSERT_EQUAL(length, data->expect_length);

    int16_t *values = (int16_t *)arr->data;

    data->samples += arr->dims[1];

    const int first_value = values[0];
    const int last_value = values[length-1];
    const int diff = last_value - first_value;

    TEST_ASSERT_EQUAL(diff, length);
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
    //EML_ARRAY_INIT_2D()

    EmlArray _input;
    EmlArray *input = &_input;
    //EML_ARRAY_INIT_2D(input);

    /* FIXME: fill each line of features with different data */
    fill_array_2d(input, 0.0, 1.0, 1.0);
    
    const EmlError init_err = eml_signal_windower_init(windower, buffer);
    TEST_ASSERT_EQUAL(init_err, EmlOk); 

    eml_signal_windower_set_callback(windower, check_windower_callback, &data);

    // TODO: test different hop lengths
    const int n_samples_input = 1;
    const int samples = 100;
    for (int i=0; i<samples; i+n_samples_input) {
        // create view
        //EML_ARRAY_INIT_2D 
        EmlArray _chunk;
        EmlArray *chunk = &_chunk;
        const EmlError add_err = eml_signal_windower_add(windower, chunk);
        TEST_ASSERT_EQUAL(add_err, EmlOk);
    }

    //TEST_ASSERT_EQUAL(data.samples, );
    //TEST_ASSERT_EQUAL(data.calls, );
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


void
test_signal_windower_1d_int16()
{
    EmlArray *arr = get_test_array_2d(1, 4, sizeof(int16_t));
    TEST_ASSERT(arr);


}


void
test_eml_signal_windower()
{
    // Add tests here
    RUN_TEST(test_signal_windower_1d_int16);
}
