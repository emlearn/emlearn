

#define EML_NET_LOG_LEVEL 1
#include <eml_trees.h>

#include <unity.h>

#define TEST_XOR_FEATURES 2
#define TEST_XOR_ROWS 4
#define TEST_BUFFER_LENGTH 10

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

void
test_eml_trees()
{
    // Add tests here
    RUN_TEST(test_trees_xor_predict);
}
