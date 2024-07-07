/*
 * emlearn example for Zephyr RTOS
 */

#include <zephyr/kernel.h>

#include "xor_model.h"

void xor_test(void)
{
    const int16_t test_inputs[4][2] = {
        { 0,    0   },
        { 255,  0   },
        { 0,    255 },
        { 255,  255 },
    };

    for (int i=0; i<4; i++) {
        const int16_t *features = test_inputs[i];
        const int out = xor_model_predict(features, 2);

        printf("xor(%d,%d) = %d\n", (int)features[0], (int)features[1], out);
    }
}

int main(void)
{
    xor_test();
    return 0;
}
