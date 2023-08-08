
#define EML_NEIGHBORS_LOG_LEVEL 1
#include <eml_neighbors.h>

#include <unity.h>

// Global variable used as backing store for the various tests


EmlError
add_items_duplicate(EmlNeighborsModel *model,
        const int16_t *item, int item_length,
        int16_t label, 
        int duplicates)
{
    for (int i=0; i<duplicates; i++) {
        EmlError add_err = eml_neighbors_add_item(model, item, item_length, label);
        if (add_err != EmlOk) {
            return add_err;
        }
    }
    return EmlOk;
} 

#define N_FEATURES 3
#define MAX_ITEMS 100
#define DATA_LENGTH (MAX_ITEMS*N_FEATURES)

void
test_neighbors_simple()
{
    // Basic check that we can discriminate between data from two different classes

    EmlError err = EmlOk;
    int16_t out_label = -1;

    // Allocate space
    int16_t data[DATA_LENGTH];
    int16_t labels[MAX_ITEMS];
    EmlNeighborsDistanceItem distances[MAX_ITEMS];

    const int K_NEIGHBORS = 1;

    // Setup model
    EmlNeighborsModel _model = { N_FEATURES, 0, MAX_ITEMS, data, labels, K_NEIGHBORS };
    EmlNeighborsModel *model = &_model;
    err = eml_neighbors_check(model, DATA_LENGTH, MAX_ITEMS, MAX_ITEMS);
    TEST_ASSERT_EQUAL(EmlOk, err);

    // Add data, class 1
    const int16_t class1_features[N_FEATURES] = { -21, -30, -20 };
    const int16_t class1_label = 0;
    err = add_items_duplicate(model, class1_features, N_FEATURES, class1_label, MAX_ITEMS-2);
    TEST_ASSERT_EQUAL(EmlOk, err);

    // Add data, class 2
    const int16_t class2_features[N_FEATURES] = { 12, 10, 10 };
    const int16_t class2_label = 1;
    err = add_items_duplicate(model, class2_features, N_FEATURES, class2_label, 2);
    TEST_ASSERT_EQUAL(EmlOk, err);

    // Predict on class 1 data
    err = eml_neighbors_predict(model, class1_features, N_FEATURES, distances, MAX_ITEMS, &out_label);
    TEST_ASSERT_EQUAL(EmlOk, err);
    TEST_ASSERT_EQUAL(class1_label, out_label);

    // Predict on class 2 data
    out_label = 31;
    err = eml_neighbors_predict(model, class2_features, N_FEATURES, distances, MAX_ITEMS, &out_label);
    TEST_ASSERT_EQUAL(EmlOk, err);
    TEST_ASSERT_EQUAL(class2_label, out_label);
}

void
test_eml_neighbors()
{
    // Add tests here
    RUN_TEST(test_neighbors_simple);
}
