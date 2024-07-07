
#include "xor_model.h" // emlearn generated model

#include <stdio.h> // printf
#include <stdlib.h> // stdod

int
main(int argc, const char *argv[])
{
    if (argc != 3) {
        return -1;
    }

    // Input is espected to be 0.0-255.0
    const float a = strtod(argv[1], NULL);
    const float b = strtod(argv[2], NULL);

    // Convert to integers
    const int16_t features[] = { a, b };

    const int out = xor_predict(features, 2);
    if (out < 0) {
        return out; // error
    } else {
        printf("%d\n", out);
    }
}
