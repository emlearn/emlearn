

#include <stdio.h>

#include "../emlearn/eml_data.h"


void data_callback(EmlDataReader *reader, const unsigned char *buffer,
                    int length, int32_t item_no)
{

    if (item_no == 0) {
        fprintf(stderr,
                "version=%d, dtype=%d, (%d, %d, %d, %d)\n",
                reader->version, reader->dtype,
                reader->dim0, reader->dim1, reader->dim2, reader->dim3);

        if (reader->dtype != EmlDataInt32) {
            fprintf(stderr, "ERROR, unexpected datatype %d\n", reader->dtype);
        }
    }

    if (reader->dtype == EmlDataInt32) {
        int x, y;
        const bool success = eml_data_reader_2dcoord(reader, item_no, &x, &y);

        if (!success) {
            fprintf(stderr, "ERROR: unable to convert coordinate\n");
            return;
        }

        const int32_t ll = eml_data_read_int32_le(buffer);
        fprintf(stderr, "data %d (x=%d, y=%d): %d | %x %x %x %x\n",
                item_no, x, y, ll,
                buffer[0], buffer[1], buffer[2], buffer[3]);

    }

}


int main() {

    FILE* file = fopen("out.emld", "r");

    const int BUFFER_LEN = 10*1000;
    char buffer[BUFFER_LEN];

    const int read = fread(buffer, 1, BUFFER_LEN, file);
   
    if (read == BUFFER_LEN) {
        fprintf(stderr, "ERROR: max file buffer size reached\n");
        return 1;
    }
    
    EmlDataReader reader;
    eml_data_reader_init(&reader);

    eml_data_reader_chunk(&reader, buffer, read, data_callback);

}
