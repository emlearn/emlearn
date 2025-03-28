
// Basic support for Comma Separated Values (CSV)
// As per RFC4180, https://www.ietf.org/rfc/rfc4180.txt
// Not intended to parse every wierd "csv" like thing out there
// Just the well-formed thing, that can be produced from typical PyData etc tools

#include <eml_common.h>

#include <stdint.h>

// I/O functions
// FIXME: standardize and share the stream definitions with eml_npy.h

typedef int (*EmlCsvReadFunction)(void *context, uint8_t *buffer, size_t size);
typedef int (*EmlCsvSeekFunction)(void *context, size_t position);
typedef int (*EmlCsvWriteFunction)(void *context, const uint8_t *buffer, size_t size);

#define EML_CSV_DELIMITER ','
#define EML_CSV_EOL '\n'

typedef struct _EmlCsvReader {

    size_t n_columns;

    // IO
    EmlCsvReadFunction read;
    void *stream;
} EmlCsvReader;


typedef struct _EmlCsvWriter {

    size_t n_columns;

    // IO
    EmlCsvWriteFunction write;
    void *stream;
} EmlCsvWriter;


void
test_eml_csv_ss() {

    #define BUFFER_LENGTH 100
    #define ROW_LENGTH 10
    float *read_data[ROW_LENGTH] = {0.0, };

    char *buffer[BUFFER_LENGTH] = {0, };

    // const char expect_columns[3] = { "a", "a_bit_longer", "b" };


    #define READ_COLUMNS_MAX 10;
    char *read_columns[READ_COLUMNS_MAX];
    err = eml_npy_reader_read_header(reader, buffer, buffer_length, read_columns, READ_COLUMNS_MAX);


}

#if 0

    "a,a_bit_longer,b\n
    1.0,1.0,1.0\n
    2.0,2.0,2.0\n
    "

#endif

// regular
// numeric. negative values, positive values
// strings
// mixed numeric and strings

// maybe
// quoted strings?
// quoted strings with comma
// literal quote character

// wierd but ok
// no newline at the end
// no string headers (straight to values)

// should error
// too few values in a row
// too many values on a row
// non-numeric data in value

// Reader
EmlError
eml_npy_reader_read_header(EmlCsvReader *self,
        char *buffer, size_t buffer_length,
        char **columns, size_t *columns_length)
{

    // TODO: maybe read byte by byte instead?
    // Copy the (potential header) into our buffer
    const size_t read_length = self->read(self->stream, buffer, buffer_length);

    // Find each column
    int column_idx = 0;
    int column_start_offset = 0;
    for (int i=0; i<read_length; i++) {

        const char c = buffer[i];
        if (c == EML_CSV_DELIMITER) {

            // Keep a pointer to start of this column
            columns[column_idx] = column_start_offset;
            column_start_offset = i+1;
            column_idx += 1;
            if (column_idx >= columns_length) {
                return EmlUnknownError;
            }
        }
    }

    // Set number of columns
    self->n_columns = column_index;

    return EmlOk;
}

EmlError
eml_npy_reader_read_data(EmlCsvReader *self,
        char *buffer, size_t buffer_length,
        char **columns, size_t *columns_length)
{
    EML_PRECONDITION(self->n_columns == data_length, EmlSizeMismatch);

    // TODO: share core implementation with read_header?
    // Means that parsing into a number stays on the outside

    //    


    return EmlOk;
}



// Writer
EmlError
eml_npy_writer_write_header(EmlCsvWriter *self, const char **columns, size_t n_columns)
{

    // Keep number of expected values per row
    self->n_columns = n_columns;

    const char *delimiter = { EML_CSV_DELIMITER };
    const char *endline = { EML_CSV_EOL };

    char buf[10] = {0,};
    for (int i=0; i<data_length; i++) {
        const char *name = columns[i];
        const int length = strlen(name);
        self->write(self->stream, name, length);
        if (i == data_length-1) {
            self->write(self->stream, delimiter, 1);
        }
    }

    self->write(self->stream, endline, 1);

    return EmlOk;
}

EmlError
eml_npy_writer_write_data(EmlCsvWriter *self, const float *data, size_t data_length)
{
    EML_PRECONDITION(self->n_columns == data_length, EmlSizeMismatch);

    const char *endline = { EML_CSV_EOL };

    char buf[10] = {0,};
    for (int i=0; i<data_length; i++) {
        int written = snprintf(buf, 10, "%f", data[i]);
        if (i == data_length-1) {
            buf[written] = EML_CSV_DELIMITER;
            written += 1;
        }
        self->write(self->stream, buf, written);
    }
    const char *endline = { EML_CSV_EOL };
    self->write(self->stream, endline, 1);

    return EmlOk;
}
