/*
 * emlearn example for Zephyr RTOS
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(main, CONFIG_APP_LOG_LEVEL);

//#define EML_TREES_REGRESSION_ENABLE 0
#include "xor_model.h"

void xor_test(void)
{
    const float test_inputs[4][2] = {
        { 0.0f, 0.0f },
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f },
    };

    for (int i=0; i<4; i++) {
        const float *features = test_inputs[i];
        const int out = xor_model_predict(features, 2);

        printf("xor(%d,%d) = %d\n", (int)features[0], (int)features[1], out);
    }
}

int main(void)
{
    xor_test();

	return 0;
}
