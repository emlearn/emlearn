
#ifndef EML_DATA_H
#define EML_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

#include "eml_common.h"

// Current version of data format
#define EML_DATA_VERSION 1

// Magic string for start
#define EML_DATA_MAGIC_LENGTH 8
static const char EML_DATA_MAGIC[EML_DATA_MAGIC_LENGTH] = "\x93EMLEARN"; 

#define EML_DATA_MAX_DIMS 4

const int EML_DATA_HEADER_LENGTH = \
    EML_DATA_MAGIC_LENGTH +
    1 + // version
    1 + // dtype
    2 + // dim0
    2 + // dim1
    2 + // dim2
    2 + // dim3
    1; // end-of-header


typedef enum _EmlDataType {
    EmlDataFloat32 = 0,
    EmlDataInt32 = 1,
} EmlDataType;

typedef struct _EmlDataReader {

    // State
    size_t byte_no; // number of bytes from start of input

    // Header data
    uint8_t version;
    EmlDataType dtype;
    uint16_t dim0;
    uint16_t dim1;
    uint16_t dim2;
    uint16_t dim3;

} EmlDataReader;

typedef void (*EmlDataReaderCallback)(EmlDataReader *reader,
                    const unsigned char *buffer, int length, int32_t item_no);


void
eml_data_reader_init(EmlDataReader *reader) {

    //reader->state = EmlDataReaderWaitMagic;
    reader->byte_no = 0;
    
    reader->version = 0;
    reader->dtype = 0;
    reader->dim0 = 0;
    reader->dim1 = 0;
    reader->dim2 = 0;
    reader->dim3 = 0;
}

uint16_t
eml_data_read_uint16_le(const unsigned char b[2]) {
    return (uint16_t)(b[0] << 8) + ((uint16_t)(b[1]) << 0); 
}

int32_t
eml_data_read_int32_le(const unsigned char b[4]) {
    const int32_t temp = ((b[0] << 24) |
            (b[1] << 16) |
            (b[2] <<  8) |
             b[3]);
    return temp;
}

float
eml_data_read_float32(const unsigned char b[4]) {
    const uint32_t temp = eml_data_read_int32_le(b);
    return *((float *) &temp);
}

bool
eml_data_reader_2dcoord(EmlDataReader *reader, int item, int *x, int *y) {

    const bool is2d = reader->dim0 != 0 && reader->dim1 != 0 &&
            reader->dim2 == 0 &&  reader->dim3 == 0;
    if (!is2d) {
        return false;
    }

    *x = (item % reader->dim0);
    *y = (item / reader->dim0);

    return true;
}

void
eml_data_reader_chunk(EmlDataReader *reader, char *input, size_t length,
                        EmlDataReaderCallback callback)
{

    unsigned char buffer[EML_DATA_HEADER_LENGTH];

    for (int i=0; i<length; i++) {
        
        if (reader->byte_no < EML_DATA_HEADER_LENGTH-1) {
            // state=in header
            buffer[reader->byte_no] = input[i];

        } else if (reader->byte_no == EML_DATA_HEADER_LENGTH-1) {
            // state=header completed

            // FIXME: check version string
            // FIXME: check the magic string

            reader->version = (uint8_t)(buffer[EML_DATA_MAGIC_LENGTH+0]);
            reader->dtype = (EmlDataType)(buffer[EML_DATA_MAGIC_LENGTH+1]);

            reader->dim0 = eml_data_read_uint16_le(buffer+(EML_DATA_MAGIC_LENGTH+2));
            reader->dim1 = eml_data_read_uint16_le(buffer+(EML_DATA_MAGIC_LENGTH+4));
            reader->dim2 = eml_data_read_uint16_le(buffer+(EML_DATA_MAGIC_LENGTH+6));
            reader->dim3 = eml_data_read_uint16_le(buffer+(EML_DATA_MAGIC_LENGTH+8));

        } else {
            // state=in data
            const int data_length = 4;
            const int data_item = (reader->byte_no - EML_DATA_HEADER_LENGTH) / data_length;
            const int item_byte = (reader->byte_no - EML_DATA_HEADER_LENGTH) % data_length;
            buffer[item_byte] = input[i];

            if (item_byte == data_length-1) {
                callback(reader, buffer, data_length, data_item);
            }
        }

        reader->byte_no += 1;
    }
}


#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_DATA_H
