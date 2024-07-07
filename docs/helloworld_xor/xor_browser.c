#include "xor_model.h" // emlearn generated model

#include <stdio.h>
#include <stdlib.h>

// Function that will be exposed to JavaScript using emscripten
int
run_xor_model(const float *features, int length)
{
    int out = -EmlSizeMismatch;
    if (length == 2) {
        // model uses 0-255 range internally
        const int16_t *quantized = { features[0]*255, features[1]*255 };
        out = xor_model_predict(quantized, length); // Alternative A: "inline"
        out = eml_trees_predict(&xor_model, quantized, length); // Alternative B: "loadable"
    } else {
        printf("run_xor_model-incorrect-length n_features=%d expected=2", length);
        return -1;
    }

    printf("run_xor_model n_features=%d inputs=(%.2f, %.2f) out=%d\n",
        length, features[0], features[1], out);
    return out;
}
