
#include "xor_model.h"

#include <stdio.h>
#include <stdlib.h> // stdod

int
main(int argc, const char *argv[])
{
    if (argc != 3) {
        return -1;
    }

    const float a = strtod(argv[1], NULL);
    const float b = strtod(argv[2], NULL);
    const float features[] = { a, b };

    const int out = xor_predict(features, 2);
    if (out < 0) {
        return out; // error
    } else {
        printf("%d\n", out);
    }
}
