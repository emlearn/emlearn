
#include <eml_array.h>

#include <unity.h>

// Global variable used as backing store for the various tests
#define TEST_ARRAY_BUFFER_SIZE 1000
uint8_t g_test_array_buffer[TEST_ARRAY_BUFFER_SIZE];

EmlArray *
get_test_array_2d(int width, int height, int value_size)
{
    static EmlArray array;
    EmlArray *arr = &array;

    int32_t dimensions[EML_ARRAY_MAX_DIMS] = { width, height, 0 };
    
    const EmlError init_err = eml_array_init_full(arr, 2, dimensions, value_size,
        g_test_array_buffer, TEST_ARRAY_BUFFER_SIZE);
    if (init_err != EmlOk) {
        return NULL;
    }
    return arr;
}

void
check_fill_sum(EmlArray *arr, float value)
{
    TEST_ASSERT_EQUAL(arr->n_dims, 2); // TODO: also support 1d and 3d
    const int dimensions = arr->dims[0] * arr->dims[1];

    float sum = -1.0;
    eml_array_fill(arr, value);
    const float expect = value*dimensions;
    TEST_ASSERT_EQUAL(eml_array_sum(arr, &sum), EmlOk);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expect, sum);
}

void
test_array_fill()
{
    EmlArray *arr = get_test_array_2d(3, 4, sizeof(int16_t));
    TEST_ASSERT(arr);

    check_fill_sum(arr, 0.0f);
    check_fill_sum(arr, 42.0f);
    // TODO: check float values
}

void
test_eml_array()
{
    // Add tests here
    RUN_TEST(test_array_fill);
}
