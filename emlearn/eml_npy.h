
#include <eml_common.h>

#include <stdint.h>

// I/O functions
typedef int (*EmlNpyReadFunction)(void *context, uint8_t *buffer, size_t size);
typedef int (*EmlNpyWriteFunction)(void *context, const uint8_t *buffer, size_t size);

typedef enum EmlNpyDtype {
    EML_NPY_F64,
    EML_NPY_F32,
    EML_NPY_U8, EML_NPY_U16, EML_NPY_U32, EML_NPY_U64,
    EML_NPY_I8, EML_NPY_I16, EML_NPY_I32, EML_NPY_I64,
    EML_NPY_UNSUPPORTED
} EmlNpyDtype;


#ifndef EML_NPY_MAX_DIMS
#define EML_NPY_MAX_DIMS 6
#endif

#ifndef EML_NPY_HEADER_BUFFER_SIZE
#define EML_NPY_HEADER_BUFFER_SIZE 64
#endif


// Header stuff
#define EML_NPY_MAGIC_STRING {-109,'N','U','M','P','Y'}
#define EML_NPY_MAGIC_LENGTH 6

#define EML_NPY_MAJOR_VERSION_IDX 6
#define EML_NPY_MINOR_VERSION_IDX 7

#define EML_NPY_VERSION_LENGTH 2
#define EML_NPY_HEADER_LENGTH_LOW_IDX 8
#define EML_NPY_HEADER_LENGTH_HIGH_IDX 9

#define EML_NPY_VERSION_HEADER_LENGTH 4
#define EML_NPY_PREHEADER_LENGTH (EML_NPY_MAGIC_LENGTH + EML_NPY_VERSION_HEADER_LENGTH)



typedef struct _EmlNpyReader {
    
    EmlNpyDtype dtype;
    size_t dims;
    size_t shape[EML_NPY_MAX_DIMS];
    void *user_data;
} EmlNpyReader;


typedef struct _EmlNpyWriter {

    EmlNpyDtype dtype;
    size_t dims;
    size_t shape[EML_NPY_MAX_DIMS];
    void *user_data;
} EmlNpyWriter;


static const char * eml_npy_dtype_to_descr(EmlNpyDtype type)
{
    switch(type)
    {
    case EML_NPY_F32:
        return "<f4";
    case EML_NPY_F64:
        return "<f8";
    case EML_NPY_I8:
        return "<i1";
    case EML_NPY_I16:
        return "<i2";
    case EML_NPY_I32:
        return "<i4";
    case EML_NPY_I64:
        return "<i8";
    case EML_NPY_U8:
        return "<u1";
    case EML_NPY_U16:
        return "<u2";
    case EML_NPY_U32:
        return "<u4";
    case EML_NPY_U64:
        return "<u8";
    case EML_NPY_UNSUPPORTED:
        return NULL;
    }
    return NULL;
}


EmlError
eml_npy_reader_init(EmlNpyReader *self)
{


}

EmlError
eml_npy_reader_read_header(EmlNpyReader *self)
{


}

EmlError
eml_npy_reader_read_data(EmlNpyReader *self, int items,
        void *buffer_data, size_t buffer_length)
{
    // FIXME: check header read OK
    // FIXME: check that buffer_length is OK

}

EmlError
eml_npy_writer_init(EmlNpyWriter *self)
{


}

EmlError
eml_npy_writer_write_header(EmlNpyWriter *self)
{
    // keep track of length, since it needs to be padded
    int length = 0;

    // Write Magic
    static char magic[] = EML_NPY_MAGIC_STRING;
    self->write(self->user_data, magic, EML_NPY_MAGIC_LENGTH);
    length += EML_NPY_MAGIC_LENGTH;

    static char version[EML_NPY_VERSION_LENGTH] = { 1, 0 };
    self->write(self->user_data, version, EML_NPY_VERSION_LENGTH );
    length += EML_NPY_VERSION_LENGTH;

    // XXX: large buffer. Maybe require API user to pass in?
    static char buffer[EML_NPY_HEADER_BUFFER_SIZE] = { '\0' };

    // First write everything before shape
    const char endianness = '<';
    const char dtype = eml_npy_dtype_to_descr(self->dtype);
    size_t out = sprintf(buffer, "{'descr': '%s', 'fortran_order': False, 'shape': (", dtype);
    self->write(self->user_data, buffer, out );

    // Write shape
    // FIXME: memset(buffer, 0); ?
    char *shape_str = buffer;
    for( int i = 0; i < self->dims; i++) {
        // FIXME: check length of buffer
        shape_str += sprintf(shape_str, "%d,", (int)self->shape[i]);
    }
    // ASSERT ( ptr - shape < EML_NPY_SHAPE_BUFSIZE );

    // Write trailing end of our shape tuple and dictionary
    const char * trailer = "), }";
    self->write(self->user_data, trailer, 4 );

    // Pad with spaces to ensure data start is aligened to 16 bytes
    int data_start = length;
    const int padding = 16-(data_start % 16)

    // FIXME: memset(buffer, 0); ?
    length += sprintf(buffer, "%*s\n", padding, " ");
    data_start = length;
    // ASSERT data_start % 16
    self->write(self->user_data, buffer, out );

    return EmlOk;
}

EmlError
eml_npy_writer_write_data(EmlNpyWriter *self, int items,
        void *buffer_data, size_t buffer_length)
{


}

#if 0

int npy_stdio_write(void *context, const uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fwrite(fptr, buffer, size);
}

int npy_stdio_read(void *context, const uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fread(fptr, buffer, size);
}

int test_file_roundtrip() {

    // write some data
    const char *generate_path = "generated.npy";
    FILE* generate_file = fopen(stream_write_path, "wb");

    // TODO: implement

    fclose(generate_file);

    const char *stream_write_path = "stream_write.npy";

    // do a streaming copy between two files
    FILE* read_file = fopen(generate_path, "rb");
    FILE* write_file = fopen(stream_write_path, "wb");

    // TODO: implement

    fclose(read_file);
    fclose(write_file);

    // read the data in

    // compare with original

}

#endif

