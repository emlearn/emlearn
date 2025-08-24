
#include <eml_csv.h>
#include <eml_fileio.h>

#include <unity.h>


#define N_DATA_COLUMNS 7
const char *columns[] = {
    "time",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
};

#define READ_BUFFER_SIZE 2*1024
char read_buffer[READ_BUFFER_SIZE];

#define READ_COLUMNS_MAX 10
char *read_columns[READ_COLUMNS_MAX];

void test_csv_file_write_readback()
{
    // Writes a CSV file, then read it back
    const char *filename = "test/out/write_readback.csv";

    // Write some simple file
    FILE *write_file = fopen(filename, "w");
    
    EmlCsvWriter _writer = {
        .n_columns = N_DATA_COLUMNS,
        .write = eml_fileio_write,
        .stream = write_file,
    };
    EmlCsvWriter *writer = &_writer;

    EmlError header_err = eml_csv_writer_write_header(writer, columns, N_DATA_COLUMNS);
    TEST_ASSERT_EQUAL(EmlOk, header_err);

    const float values[2*N_DATA_COLUMNS] = {
        0.0f, 1.1f, 2.2f, -3356.910f, 4.4f, -5.5f, 666.666f,
        1.0f, 99.1f, 2.2f, -3.14f, 4.4f, -5.5f, -6.6f,
    };

    EmlError write_err = EmlUnknownError;
    const int write_rows = 2;
    for (int i=0; i<write_rows; i++) {
        const float *row = values + (i * N_DATA_COLUMNS);
        write_err = eml_csv_writer_write_data(writer, row, N_DATA_COLUMNS);
        TEST_ASSERT_EQUAL(EmlOk, write_err);
    }
    printf("write-done header=%d write=%d \n", header_err, write_err);
    fclose(write_file);
   

    // Read back
    FILE *read_file = fopen(filename, "r");
    EmlCsvReader _reader = {
        .seek = eml_fileio_seek,
        .read = eml_fileio_read,
        .stream = read_file,
    };
    EmlCsvReader *reader = &_reader;

    EmlError read_header_err = eml_csv_reader_read_header(reader,\
        read_buffer, READ_BUFFER_SIZE, read_columns, READ_COLUMNS_MAX);
    TEST_ASSERT_EQUAL(EmlOk, read_header_err);

#if 0
    printf("read-columns: \n");
    for (int i=0; i<reader->n_columns; i++) {
        printf("%s\n", read_columns[i]);
    }
#endif

    const int expect_columns = N_DATA_COLUMNS;
    TEST_ASSERT_EQUAL(expect_columns, reader->n_columns);

    // TODO: read all rows, check got everything
    int read_values;

    for (int row=0; row<100; row++) {
        read_values = eml_csv_reader_read_data(reader,\
            read_buffer, READ_BUFFER_SIZE, read_columns, READ_COLUMNS_MAX);
        if (read_values == 0) {
            break;
        }
        TEST_ASSERT_EQUAL(expect_columns, read_values);

        for (int v=0; v<reader->n_columns; v++) {
            const float got = strtod(read_columns[v], NULL);
            const float expect = values[(row*N_DATA_COLUMNS)+v];
            printf("read-value-compare expect=%f got=%f\n", expect, got);
            TEST_ASSERT_FLOAT_WITHIN(0.01f, expect, got);
        }
    }

}


void
test_eml_csv_file()
{
    // Add tests here
    RUN_TEST(test_csv_file_write_readback);
}
