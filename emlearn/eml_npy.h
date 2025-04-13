
#include <eml_common.h>

#include <stdint.h>

// I/O functions
typedef int (*EmlNpyReadFunction)(void *context, uint8_t *buffer, size_t size);
typedef int (*EmlNpySeekFunction)(void *context, size_t position);
typedef int (*EmlNpyWriteFunction)(void *context, const uint8_t *buffer, size_t size);

typedef enum EmlNpyDtype {
    EML_NPY_F64,
    EML_NPY_F32,
    EML_NPY_U8, EML_NPY_U16, EML_NPY_U32, EML_NPY_U64,
    EML_NPY_I8, EML_NPY_I16, EML_NPY_I32, EML_NPY_I64,
    EML_NPY_UNSUPPORTED
} EmlNpyDtype;


#ifndef EML_NPY_MAX_DIMS
#define EML_NPY_MAX_DIMS 4
#endif

#ifndef EML_NPY_HEADER_BUFFER_SIZE
#define EML_NPY_HEADER_BUFFER_SIZE 128
#endif


// Header stuff
#define EML_NPY_MAGIC_STRING {-109,'N','U','M','P','Y'}
#define EML_NPY_MAGIC_LENGTH 6

#define EML_NPY_MAJOR_VERSION_IDX 6
#define EML_NPY_MINOR_VERSION_IDX 7
#define EML_NPY_VERSION_LENGTH 2

#define EML_NPY_HEADER_LENGTH_LOW_IDX 8
#define EML_NPY_HEADER_LENGTH_HIGH_IDX 9
#define EML_NPY_SIZE_HEADER_LENGTH 2

#define EML_NPY_PREHEADER_LENGTH (EML_NPY_MAGIC_LENGTH + EML_NPY_VERSION_LENGTH + EML_NPY_SIZE_HEADER_LENGTH)



typedef struct _EmlNpyReader {
    
    EmlNpyDtype dtype;
    size_t dims;
    size_t shape[EML_NPY_MAX_DIMS];

    size_t data_start;

    EmlNpyReadFunction read;
    EmlNpySeekFunction seek;
    void *stream;
} EmlNpyReader;


typedef struct _EmlNpyWriter {

    EmlNpyDtype dtype;
    size_t dims;
    size_t shape[EML_NPY_MAX_DIMS];

    EmlNpyWriteFunction write;
    EmlNpySeekFunction seek;
    void *stream;
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


int
eml_npy_dtype_size(EmlNpyDtype dtype)
{
    switch(dtype)
    {
    case EML_NPY_F32:
        return 4;
    case EML_NPY_F64:
        return 8;
    case EML_NPY_I8:
        return 1;
    case EML_NPY_I16:
        return 2;
    case EML_NPY_I32:
        return 4;
    case EML_NPY_I64:
        return 8;
    case EML_NPY_U8:
        return 1;
    case EML_NPY_U16:
        return 2;
    case EML_NPY_U32:
        return 4;
    case EML_NPY_U64:
        return 8;
    case EML_NPY_UNSUPPORTED:
        return 0;
    }
    return 0;
}

int
eml_npy_compute_items(const size_t *shape, size_t dims)
{
    int items = 1;
    for (int i=0; i<dims; i++) {
        items *= shape[i];
    }
    return items;
}
int
eml_npy_compute_size(const size_t *shape, size_t dims, EmlNpyDtype dtype)
{
    const int items = eml_npy_compute_items(shape, dims);
    const int item_size = eml_npy_dtype_size(dtype);
    const int bytes = item_size * items;
    return bytes;
}



EmlError
eml_npy_reader_init(EmlNpyReader *self)
{


}

static char *
eml_npy_find_header_item( const char *item, const char *header)
{
    char *s = strstr(header, item);
    return s ? s + strlen(item) : NULL;
}

EmlError
eml_npy_reader_read_header(EmlNpyReader *self)
{
    // TODO: parse the header
    // set dtype, shape and data_start

    // XXX: large buffer. Maybe require API user to pass in?
    // XXX: this could be split into two reads, one for the pre-header, and then the specified bytes
    static char buffer[EML_NPY_HEADER_BUFFER_SIZE] = { '\0' };
    const size_t buffer_length = EML_NPY_HEADER_BUFFER_SIZE;

    // Read fixed-length portion of header
    const size_t read_length = self->read(self->stream, buffer, EML_NPY_PREHEADER_LENGTH);
    if (read_length != EML_NPY_PREHEADER_LENGTH) {
        return EmlUnknownError;
    }

    // Check magic
    static char magic[] = EML_NPY_MAGIC_STRING;
    for (int i = 0; i < EML_NPY_MAGIC_LENGTH; i++ ){
        if( magic[i] != buffer[i] ){
            return EmlUnsupported;
        }
    }

    // Check version
    char major_version = buffer[EML_NPY_MAJOR_VERSION_IDX];
    char minor_version = buffer[EML_NPY_MINOR_VERSION_IDX];
    if (major_version != 1) {
        return EmlUnsupported;
    }

    // Check length - is to be stored little-endian
    // XXX: version=1 only
    uint16_t header_length = buffer[EML_NPY_HEADER_LENGTH_LOW_IDX] + \
        (buffer[EML_NPY_HEADER_LENGTH_HIGH_IDX] << 8);
    if (header_length > buffer_length) {
        return EmlSizeMismatch;
    }

    // Read the variable part of header
    const size_t header_read = self->read(self->stream, buffer, header_length);
    if (header_read != header_length) {
        return EmlUnknownError;
    }
    char *header = buffer;
    header[header_length] = '\0';

    const char *descr = eml_npy_find_header_item("'descr': '", header);
    if (!descr) {
        return EmlUnknownError;
    }
    // ASSERT

    if (strchr("<>|", descr[0])) {
        const char endianness = descr[0];
        if (descr[0] != '|' && ( descr[0] != '<')) {
            return EmlUnsupported;
        }
    } else {
        return EmlUnknownError; // Invalid
    }

    // Parse dtype
    //m->typechar = descr[1];
    //m->elem_size = (size_t) strtoll( &descr[2], NULL, 10);

    // Parse data order
    // XXX: assumes one leading space
    char *order = eml_npy_find_header_item("'fortran_order': ", header);
    if (!order) {
        return EmlUnknownError;
    }

    if (strncmp(order, "True", 4) == 0 ) {
        return EmlUnsupported;
    } else if (strncmp(order, "False", 5) == 0 ) {
        // C order, OK
    } else {
        return EmlUnknownError; // Invalid
    }

    // Parse shape
    // XXX: assumes one leading space
    char *shape = eml_npy_find_header_item("'shape': ", header);
    //FIXME: assert(shape);
    
    while (*shape != ')' ) {
#if 0
        if( !isdigit( (int) *shape ) ){
            shape++;
            continue;
        }
        self->shape[m->ndim] = strtol( shape, &shape, 10);
        ndim++;
        assert( m->ndim < EML_NPY_MAX_DIMENSIONS );
#endif
    }

    // Compute start of data
    self->data_start = EML_NPY_PREHEADER_LENGTH + header_length;

    // Seek to start of data by default
    self->seek(self->stream, self->data_start);
    return EmlOk;
}

EmlError
eml_npy_reader_read_data(EmlNpyReader *self, int items,
        void *buffer_data, size_t buffer_length, int *out_items)
{
    const size_t expect_bytes = eml_npy_compute_size(self->shape, self->dims, self->dtype);
    EML_PRECONDITION(buffer_length >= expect_bytes, EmlSizeMismatch);

    EML_PRECONDITION(self->data_start > 0, EmlUninitialized);

    const size_t bytes_read = self->read(self->stream, buffer_data, expect_bytes);
    const int item_size = eml_npy_dtype_size(self->dtype);
    if ((bytes_read % item_size) != 0) {
        return EmlUnknownError;
    }
    const int items_read = bytes_read/item_size;
    *out_items = items_read;

    return EmlOk;
}

EmlError
eml_npy_reader_seek(EmlNpyReader *self, size_t item)
{
    EML_PRECONDITION(self->data_start > 0, EmlUninitialized);

    const size_t bytes = eml_npy_compute_size(self->shape, self->dims, self->dtype);
    const size_t position = self->data_start + bytes;
    const int seek_status = self->seek(self->stream, position);
    // FIXME: check seek_status

    return EmlOk;
}

EmlError
eml_npy_writer_init(EmlNpyWriter *self)
{


}

EmlError
eml_npy_writer_write_header(EmlNpyWriter *self, EmlNpyDtype dtype, const size_t *shape, size_t dims)
{
    EML_PRECONDITION(dims <= EML_NPY_MAX_DIMS, EmlUnsupported);

    // Copy header information
    self->dtype = dtype;
    self->dims = dims;
    for (int i=0; i<dims; i++) {
        self->shape[i] = shape[i];
    }

    // keep track of length, since it needs to be padded
    int length = 0;

    // Write Magic
    static char magic[] = EML_NPY_MAGIC_STRING;
    self->write(self->stream, magic, EML_NPY_MAGIC_LENGTH);
    length += EML_NPY_MAGIC_LENGTH;

    static char version[EML_NPY_VERSION_LENGTH] = { 1, 0 };
    self->write(self->stream, version, EML_NPY_VERSION_LENGTH );
    length += EML_NPY_VERSION_LENGTH;

    // XXX: large buffer. Maybe require API user to pass in?
    static char buffer[EML_NPY_HEADER_BUFFER_SIZE] = { '\0' };

    // First write everything before shape
    const char endianness = '<';
    const char *dtype_str = eml_npy_dtype_to_descr(self->dtype);
    size_t out = sprintf(buffer, "{'descr': '%s', 'fortran_order': False, 'shape': (", dtype_str);
    self->write(self->stream, buffer, out );

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
    self->write(self->stream, trailer, 4 );

    // Pad with spaces to ensure data start is aligened to 16 bytes
    int data_start = length;
    const int padding = 16-(data_start % 16);

    // FIXME: memset(buffer, 0); ?
    length += sprintf(buffer, "%*s\n", padding, " ");
    data_start = length;
    // ASSERT data_start % 16
    self->write(self->stream, buffer, out );

    return EmlOk;
}

EmlError
eml_npy_writer_write_data(EmlNpyWriter *self, int items,
        void *buffer_data, size_t buffer_length)
{
    // FIXME: check buffer_length wrt expected based on items
    

}


