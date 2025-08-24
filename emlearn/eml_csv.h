
// Basic support for Comma Separated Values (CSV)
// As per RFC4180, https://www.ietf.org/rfc/rfc4180.txt
// Not intended to parse every wierd "csv" like thing out there
// Just the well-formed thing, that can be produced from typical PyData etc tools

#include <eml_common.h>
#include <eml_io.h>

#include <stdint.h>
#include <string.h>


#define EML_CSV_DELIMITER ","
#define EML_CSV_EOL "\n"

typedef struct _EmlCsvReader {

    size_t n_columns;

    // IO
    EmlIoReadFunction read;
    EmlIoSeekFunction seek;
    void *stream;
} EmlCsvReader;


typedef struct _EmlCsvWriter {

    size_t n_columns;

    // IO
    EmlIoWriteFunction write;
    EmlIoSeekFunction seek;
    void *stream;
} EmlCsvWriter;


// Writer
EmlError
eml_csv_writer_write_header(EmlCsvWriter *self, const char **columns, size_t n_columns)
{

    // Keep number of expected values per row
    self->n_columns = n_columns;
    const int data_length = n_columns;

    const char *delimiter = EML_CSV_DELIMITER;
    const char *endline = EML_CSV_EOL;

    char buf[10] = {0,};
    for (int i=0; i<data_length; i++) {
        const char *name = columns[i];
        const int length = strlen(name);
        self->write(self->stream, name, length);
        const bool is_last = (i == data_length-1);
        if (!is_last) {
            self->write(self->stream, delimiter, 1);
        }
    }

    self->write(self->stream, endline, 1);

    return EmlOk;
}

EmlError
eml_csv_writer_write_data(EmlCsvWriter *self, const float *data, size_t data_length)
{
    EML_PRECONDITION(self->n_columns == data_length, EmlSizeMismatch);

    const char *endline = EML_CSV_EOL;

    char buf[10] = {0,};
    for (int i=0; i<data_length; i++) {
        int written = snprintf(buf, 10, "%f", data[i]);
        const bool is_last = (i == data_length-1);
        if (!is_last) {
            buf[written] = (EML_CSV_DELIMITER)[0];
            written += 1;
        }
        self->write(self->stream, buf, written);
    }
    self->write(self->stream, endline, 1);

    return EmlOk;
}

// Reader

EmlError
eml_csv_reader_read_internal(EmlCsvReader *self,
        char *buffer, size_t buffer_length,
        char **columns, size_t columns_length,
        int *out_columns, int *out_offset
        )
{

    // Copy the (potential header) into our buffer
    // We will return pointers into this buffer
    const size_t read_length = self->read(self->stream, buffer, buffer_length);

    // Find start of data / end-of-header
    int data_start = -1;

    // Find each column
    int column_idx = 0;
    int column_start_offset = 0;
    int offset = -1;
    for (offset=0; offset<read_length; offset++) {

        const char c = buffer[offset];
        if (c == (EML_CSV_DELIMITER)[0]) {
            // Keep a pointer to start of this column
            columns[column_idx] = buffer+column_start_offset;
            column_start_offset = offset+1;
            column_idx += 1;

            // NULL terminate previous string
            buffer[offset] = '\0';
            if (column_idx >= columns_length) {
                printf("parse-too-many-columns read=%d max=%d \n", column_idx, columns_length);
                return EmlUnknownError;
            }
        } else if (c == (EML_CSV_EOL)[0]) {
            buffer[offset] = '\0';
            break;
        }
    }

    *out_columns = column_idx;
    *out_offset = offset;

    return EmlOk;
}


EmlError
eml_csv_reader_read_header(EmlCsvReader *self,
        char *buffer, size_t buffer_length,
        char **columns, size_t columns_length)
{
    int columns_read = 0;
    int characters_read = 0;

    EmlError read_err = eml_csv_reader_read_internal(self, \
        buffer, buffer_length, columns, columns_length, &columns_read, &characters_read);

    // Rembember number of columns
    self->n_columns = columns_read;

    // Seek to start of data
    const int seek_status = self->seek(self->stream, characters_read+1);
    if (seek_status != 0) {
        return EmlUnknownError;
    }
    return EmlOk;
}

EmlError
eml_csv_reader_read_data(EmlCsvReader *self,
        char *buffer, size_t buffer_length,
        char **columns, size_t columns_length)
{
    EML_PRECONDITION(columns_length >= self->n_columns, EmlSizeMismatch);

    int columns_read = 0;
    int characters_read = 0;

    EmlError read_err = eml_csv_reader_read_internal(self, \
        buffer, buffer_length, columns, columns_length, &columns_read, &characters_read);
    EML_CHECK_ERROR(read_err);

    if (columns_read != self->n_columns) {
        printf("uncorrect-columns read=%d \n", columns_read);
        return EmlUnknownError;
    }

    // Seek to start of next line
    const int seek_status = self->seek(self->stream, characters_read+1);
    if (seek_status != 0) {
        return EmlUnknownError;
    }

    return EmlOk;
}

