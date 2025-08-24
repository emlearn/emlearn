
#include <eml_csv.h>
#include <eml_fileio.h>

#include <unity.h>


void
test_trees_xor_predict()
{
    // Basic check that we can discriminate between data from two different classes
    EmlError err = EmlOk;
    int16_t out_label = -1;

    float buffer1[TEST_BUFFER_LENGTH];
    float buffer2[TEST_BUFFER_LENGTH];

    // Setup XOR model, single decision tree
    int32_t roots[1] = { 0 };
    uint8_t leaves[2] = { 0, 1 };
    EmlTreesNode nodes[] = {
        // feature, value, left, right
        { 0, 0, 1, 2 },
        { 1, 0, -1, -2 },
        { 1, 0, -2, -1 },
    };
    EmlTrees _model;
    EmlTrees *model = &_model;

    model->n_nodes = 3;
    model->nodes = nodes;
    model->n_trees = 1;
    model->tree_roots = roots;
    model->n_leaves = 2;
    model->leaves = leaves;
    model->leaf_bits = 0; // majority voting
    model->n_features = 2;
    model->n_classes = 2;

    // Test data
    int16_t test_data[TEST_XOR_ROWS][TEST_XOR_FEATURES+1] = {
        // in1, in2 -> out 
        { -1, -1,  0 },
        {  1,  1,  0 },
        { -1,  1,  1 },
        {  1, -1,  1 },
    };

    float out[2];

    for (int i=0; i<TEST_XOR_ROWS; i++) {
        const int16_t *data = test_data[i];
        const int16_t expect_class = data[2];   
        const float expect_proba0 = (expect_class == 0) ? 1.0f : 0.0f;
        const float expect_proba1 = (expect_class == 0) ? 0.0f : 1.0f;

        printf("test-xor case=%d in=[%d %d] expect=%d\n", i, data[0], data[1], expect_class);

        const EmlError err = eml_trees_predict_proba(model, data, TEST_XOR_FEATURES, out, 2);
        TEST_ASSERT_EQUAL(EmlOk, err);
        TEST_ASSERT_FLOAT_WITHIN(0.01f, expect_proba0, out[0]);
        TEST_ASSERT_FLOAT_WITHIN(0.01f, expect_proba1, out[1]);

        const int out = eml_trees_predict(model, data, TEST_XOR_FEATURES);
        TEST_ASSERT_EQUAL(out, expect_class);
    }
}



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

#define READ_BUFFER_SIZE 1024
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

    const float values[N_DATA_COLUMNS] = \
        { 0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f };

    EmlError write_err = EmlUnknownError;
    const int write_rows = 2;
    for (int i=0; i<write_rows; i++) {
        write_err = eml_csv_writer_write_data(writer, values, N_DATA_COLUMNS);
        TEST_ASSERT_EQUAL(EmlOk, write_err);
    }

    write_err = eml_csv_writer_write_data(writer, values, N_DATA_COLUMNS);
    TEST_ASSERT_EQUAL(EmlOk, write_err);

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

    printf("header-status err=%d columms=%d\n", read_header_err, reader->n_columns);

    const int expect_columns = N_DATA_COLUMNS;
    TEST_ASSERT_EQUAL(expect_columns, reader->n_columns);

    // TODO: check that columns are correct
    printf("columns: \n");
    for (int i=0; i<reader->n_columns; i++) {
        printf("%s\n", read_columns[i]);
    }

    // TODO: read all rows, check got everything
    EmlError data_err = eml_csv_reader_read_data(reader,\
        read_buffer, READ_BUFFER_SIZE, read_columns, READ_COLUMNS_MAX);
    TEST_ASSERT_EQUAL(EmlOk, data_err);

    printf("data-status err=%d\n", data_err);
    for (int i=0; i<reader->n_columns; i++) {
        const float v = strtod(read_columns[i], NULL);
        printf("%s %f\n", read_columns[i], v);
    }

}


void
test_eml_csv_file()
{
    // Add tests here
    RUN_TEST(test_csv_file_write_readback);
}
