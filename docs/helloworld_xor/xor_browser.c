#include "xor_model.h" // emlearn generated model

#include <stdio.h>
#include <stdlib.h>

// Function that will be exposed to JavaScript using emscripten
int
run_xor_model(const float *features, int length)
{
    int out = -EmlSizeMismatch;
    float a = -1.0;
    float b = -1.0;
    if (length == 2) {
        a = features[0];
        b = features[1];
        out = xor_model_predict(features, length); // Alternative A: "inline"
        out = eml_trees_predict(&xor_model, features, length); // Alternative B: "loadable"
    }

    printf("run_xor_model n_features=%d inputs=(%.2f, %.2f) out=%d\n", length, a, b, out);
    return out; 
}
