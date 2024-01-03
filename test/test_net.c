
#define EML_NET_LOG_LEVEL 1
#include <eml_net.h>

#include <unity.h>

#define TEST_NET_N_FEATURES 1
#define TEST_BUFFER_LENGTH 10

void
test_net_logreg_binary()
{
    // Basic check that we can discriminate between data from two different classes
    EmlError err = EmlOk;
    int16_t out_label = -1;

    float buffer1[TEST_BUFFER_LENGTH];
    float buffer2[TEST_BUFFER_LENGTH];

    const float layer1_weigths[] = { 1.0f };
    const float layer1_biases[] = { 0.0f };

    const float layer2_weigths[] = { 1.0f };
    const float layer2_biases[] = { 0.0f };
    const EmlNetLayer layers[] = {
        { 1, 1, layer1_weigths, layer1_biases, EmlNetActivationIdentity },
        { 1, 1, layer2_weigths, layer2_biases, EmlNetActivationLogistic }
    };

    // Test data
    float class1_features[TEST_NET_N_FEATURES] = { -0.2 };
    float class2_features[TEST_NET_N_FEATURES] = { 0.2 };

    // Setup model
    EmlNet _model = { 2, layers, buffer1, buffer2, TEST_BUFFER_LENGTH };
    EmlNet *model = &_model;
    const bool model_valid = eml_net_valid(model);
    TEST_ASSERT_TRUE(model_valid);

    // Predict on class 1 data
    out_label = eml_net_predict(model, class1_features, TEST_NET_N_FEATURES);
    TEST_ASSERT_EQUAL(0, out_label);

    // Predict on class 2 data
    out_label = eml_net_predict(model, class2_features, TEST_NET_N_FEATURES);
    TEST_ASSERT_EQUAL(1, out_label);
}

void
test_eml_net()
{
    // Add tests here
    RUN_TEST(test_net_logreg_binary);
}
