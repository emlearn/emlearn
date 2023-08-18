
#include <eml_quantizer.h>

#include <unity.h>




void
test_quantizer_simple()
{
    // Basic check that we can discriminate between data from two different classes

    EmlError err = EmlOk;

#define N_FEATURES 5
    // Values above 1000 will be truncated
    EmlQuantizer quantizer = { 32767.0/1000.0 };
    // Input has one value above range, and one below range
    float in[N_FEATURES] = { -1234.0, -120.0, 23.0, -32.0, 12999.0 }; 
    int16_t out[N_FEATURES];
    float inv[N_FEATURES];

    int underflows = -1;
    int overflows = -1;
    err = eml_quantizer_check_forward_int16(&quantizer, in, N_FEATURES, out, N_FEATURES, &underflows, &overflows);
    TEST_ASSERT_EQUAL(EmlOk, err);

    // check that over/underflows are caight
    TEST_ASSERT_EQUAL(1, overflows);
    TEST_ASSERT_EQUAL(1, underflows);

    // other values should be approximately equal when inverted
    err = eml_quantizer_inverse_int16(&quantizer, out, N_FEATURES, inv, N_FEATURES);
    const float max_delta = 0.10;
    TEST_ASSERT_FLOAT_WITHIN(max_delta, in[1], inv[1]);
    TEST_ASSERT_FLOAT_WITHIN(max_delta, in[2], inv[2]);
    TEST_ASSERT_FLOAT_WITHIN(max_delta, in[3], inv[3]);

#undef N_FEATURES
}

void
test_eml_quantizer()
{
    // Add tests here
    RUN_TEST(test_quantizer_simple);
}
