
#include <eml_npy.h>

#include <unity.h>

// TODO: maybe move to a general header?
int eml_npy_stdio_write(void *context, const uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fwrite(buffer, 1, size, fptr);
}

int eml_npy_stdio_read(void *context, uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fread(buffer, 1, size, fptr);
}

int eml_npy_stdio_seek(void *context, size_t position)
{
    FILE* fptr = context;
    return fseek(fptr, position, SEEK_SET);
}

// Fill buffer with consecutive numbers
int
generate_arange(void *buffer, size_t buffer_length,
        const size_t *shape, size_t dims,
        EmlNpyDtype dtype, int start)
{
    const int expect_bytes = eml_npy_compute_size(shape, dims, dtype);    

    if (buffer_length < expect_bytes) {
        return -1;
    } 

    int item = 0;
    int next = start;
    for (int d=0; d<dims; d++) {
        for (int i=0; i<shape[d]; i++) {
            next += 1;

            if (dtype == EML_NPY_I32) {
                int32_t *data = buffer;
                data[item] = next;
            } else if (dtype == EML_NPY_U8) {
                uint8_t *data = buffer;
                data[item] = next;
            } else {
                return -2;
            }
            item += 1;
        }
    }

    // return actual length written
    const int item_size = eml_npy_dtype_size(dtype);
    const int out_bytes = item_size * item;
    return out_bytes;
}


void
check_file_roundtrip(const size_t *shape, size_t dims, EmlNpyDtype dtype, int start)
{
    EmlNpyWriter _writer;
    EmlNpyWriter *writer = &_writer;
    EmlError err = EmlOk;

    EmlNpyReader _reader;
    EmlNpyReader *reader = &_reader;

    // generate some data
#define GENERATE_DATA_MAXLENGTH 512
    static uint8_t generate_buffer[GENERATE_DATA_MAXLENGTH];
    const int generate_bytes = \
        generate_arange(generate_buffer, GENERATE_DATA_MAXLENGTH, shape, dims, dtype, start);
    const int generate_items = eml_npy_compute_items(shape, dims);
#undef GENERATE_DATA_MAXLENGTH

    // write generated data to file
    const char *generate_path = "generated.npy";
    FILE* generate_file = fopen(generate_path, "wb");

    writer->write = eml_npy_stdio_write;
    writer->seek = eml_npy_stdio_seek;
    writer->stream = (void *)generate_file;
    err = eml_npy_writer_write_header(writer, dtype, shape, dims);
    TEST_ASSERT_EQUAL(EmlOk, err);
    err = eml_npy_writer_write_data(writer, generate_items, generate_buffer, generate_bytes);
    TEST_ASSERT_EQUAL(EmlOk, err);
    fclose(generate_file);

#if 0
    // do a streaming copy between two files
    const char *stream_write_path = "stream_write.npy";
    FILE* read_file = fopen(generate_path, "rb");
    FILE* write_file = fopen(stream_write_path, "wb");


    // TODO: implement

    fclose(read_file);
    fclose(write_file);
#endif

    // read the data in

    // compare with original

}


typedef struct {
    size_t shape[EML_NPY_MAX_DIMS];
    size_t dims;
    EmlNpyDtype dtype;
    int start;
} TestNpyFileRoundtripData;

#define ROUNDTRIP_CASES 2
TestNpyFileRoundtripData roundtrip_cases[ROUNDTRIP_CASES] = {
    { { 2, 3, 0, 0 }, 2, EML_NPY_I32, -100 },
    { { 2, 3, 0, 0 }, 2, EML_NPY_U8, 0 },
};

void
test_npy_file_roundtrips()
{
    for (int i=0; i<ROUNDTRIP_CASES; i++) {
        TestNpyFileRoundtripData data = roundtrip_cases[i];
        check_file_roundtrip(data.shape, data.dims, data.dtype, data.start);
    }

}

void
test_eml_npy_file()
{
    // Add tests here
    RUN_TEST(test_npy_file_roundtrips);
}

