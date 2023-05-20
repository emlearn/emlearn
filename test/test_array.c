
#include <eml_array.h>

#include <unity.h>

// Global variable used as backing store for the various tests
#define TEST_ARRAY_BUFFER_SIZE 1000
uint8_t g_test_array_buffer_1[TEST_ARRAY_BUFFER_SIZE];
uint8_t g_test_array_buffer_2[TEST_ARRAY_BUFFER_SIZE];

EmlArray *
get_test_array_2d(int width, int height, int value_size)
{
    static EmlArray array;
    EmlArray *arr = &array;

    int32_t dimensions[EML_ARRAY_MAX_DIMS] = { width, height, 0 };
    
    const EmlError init_err = eml_array_init_full(arr, 2, dimensions, value_size,
        g_test_array_buffer_1, TEST_ARRAY_BUFFER_SIZE);
    if (init_err != EmlOk) {
        return NULL;
    }
    return arr;
}

EmlArray *
get_test_array_2d_2(int width, int height, int value_size)
{
    static EmlArray array;
    EmlArray *arr = &array;

    int32_t dimensions[EML_ARRAY_MAX_DIMS] = { width, height, 0 };
    
    const EmlError init_err = eml_array_init_full(arr, 2, dimensions, value_size,
        g_test_array_buffer_2, TEST_ARRAY_BUFFER_SIZE);
    if (init_err != EmlOk) {
        return NULL;
    }
    return arr;
}

EmlError
fill_array_2d(EmlArray *arr, float start, float x_increment, float y_increment)
{
    EML_PRECONDITION(arr->n_dims == 2, EmlSizeMismatch);

    const int columns = arr->dims[0];
    const int rows = arr->dims[1];

    for (int i=0; i<rows; i++) {
        float value = (y_increment*i) + start;

        for (int j=0; j<columns; j++) {
            // FIXME: support floating-point
            int16_t *data = (int16_t *)eml_array_data_2d(arr, j, i);
            *data = (int16_t)value;

            value += x_increment;
        }

    }

    return EmlOk;
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

int16_t
array_2d_get_i16(EmlArray *arr, int x, int y)
{
    int16_t *data = (int16_t *)eml_array_data_2d(arr, x, y);
    TEST_ASSERT(data);
    return *data;
}

void
test_array_order_2d()
{
    EmlArray *arr = get_test_array_2d(2, 2, sizeof(int16_t));
    TEST_ASSERT(arr);

    fill_array_2d(arr, 0.0, 1.0, 10.0);
    TEST_ASSERT_EQUAL_INT(0.0, array_2d_get_i16(arr, 0, 0));
    TEST_ASSERT_EQUAL_INT(1.0, array_2d_get_i16(arr, 1, 0));
    TEST_ASSERT_EQUAL_INT(10.0, array_2d_get_i16(arr, 0, 1));
    TEST_ASSERT_EQUAL_INT(11.0, array_2d_get_i16(arr, 1, 1));
}

void
test_array_shift_2d()
{
    EmlArray *arr = get_test_array_2d(2, 4, sizeof(int16_t));
    TEST_ASSERT(arr);

    fill_array_2d(arr, 0.0, 1.0, 10.0);
    // sanity check
    TEST_ASSERT_EQUAL_INT(0.0, array_2d_get_i16(arr, 0, 0));
    TEST_ASSERT_EQUAL_INT(1.0, array_2d_get_i16(arr, 1, 0));

    // shift by 0 does nothing
    eml_array_shift_rows(arr, 0);
    TEST_ASSERT_EQUAL_INT(0.0, array_2d_get_i16(arr, 0, 0));
    TEST_ASSERT_EQUAL_INT(1.0, array_2d_get_i16(arr, 1, 0));

    // shift with negative moves data towards lower row numbers
    TEST_ASSERT_EQUAL_INT(20.0, array_2d_get_i16(arr, 0, 2));
    TEST_ASSERT_EQUAL_INT(21.0, array_2d_get_i16(arr, 1, 2));
    eml_array_shift_rows(arr, -2);
    TEST_ASSERT_EQUAL_INT(20.0, array_2d_get_i16(arr, 0, 0));
    TEST_ASSERT_EQUAL_INT(21.0, array_2d_get_i16(arr, 1, 0));

    // forward shifts currently unsupported
    TEST_ASSERT_EQUAL(eml_array_shift_rows(arr, 2), EmlUnsupported);
}

void
test_array_copy_rows()
{
    EmlError err = EmlOk;

    EmlArray *arr = get_test_array_2d(2, 5, sizeof(int16_t));
    TEST_ASSERT(arr);

    EmlArray *arr2 = get_test_array_2d_2(2, 2, sizeof(int16_t));
    TEST_ASSERT(arr2);

    // copying into start
    fill_array_2d(arr, 0.0, 1.0, 10.0);
    fill_array_2d(arr2, 5.0, 1.0, 10.0);
    err = eml_array_copy_rows(arr, 0, arr2);
    TEST_ASSERT_EQUAL(err, EmlOk);
    TEST_ASSERT_EQUAL_INT(5.0, array_2d_get_i16(arr, 0, 0));
    TEST_ASSERT_EQUAL_INT(6.0, array_2d_get_i16(arr, 1, 0));
    TEST_ASSERT_EQUAL_INT(15.0, array_2d_get_i16(arr, 0, 1));
    TEST_ASSERT_EQUAL_INT(16.0, array_2d_get_i16(arr, 1, 1));

    // copying into middle
    fill_array_2d(arr, 0.0, 1.0, 10.0);
    fill_array_2d(arr2, 5.0, 1.0, 10.0);
    err = eml_array_copy_rows(arr, 2, arr2);
    TEST_ASSERT_EQUAL(err, EmlOk);
    TEST_ASSERT_EQUAL_INT(5.0, array_2d_get_i16(arr, 0, 2));
    TEST_ASSERT_EQUAL_INT(6.0, array_2d_get_i16(arr, 1, 2));
    TEST_ASSERT_EQUAL_INT(15.0, array_2d_get_i16(arr, 0, 3));
    TEST_ASSERT_EQUAL_INT(16.0, array_2d_get_i16(arr, 1, 3));

    // another size of columns should error
    EmlArray *other_size = get_test_array_2d_2(1, 2, sizeof(int16_t));
    TEST_ASSERT(other_size);
    EmlError other_size_err = eml_array_copy_rows(arr, 0, other_size);
    TEST_ASSERT_EQUAL(other_size_err, EmlSizeMismatch);
}

void
test_eml_array()
{
    // Add tests here
    RUN_TEST(test_array_fill);
    RUN_TEST(test_array_order_2d);
    RUN_TEST(test_array_shift_2d);
    RUN_TEST(test_array_copy_rows);
}
