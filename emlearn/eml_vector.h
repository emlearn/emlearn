
#ifndef EMVECTOR_H
#define EMVECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "eml_common.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    float *data;
    int length;
} EmlVector;


int
eml_vector_set(EmlVector dest, EmlVector source, int location) {
    const int final_dest = source.length+location;
    if (final_dest > dest.length) {
        return -1;
    }
    if (location < 0) {
        return -2;
    }

    for (int i=location; i<final_dest; i++) {
        dest.data[i] = source.data[i-location]; 
    }
    return 0;
}


EmlError
eml_vector_set_value(EmlVector a, float val) {
    for (int32_t i=0; i<a.length; i++) {
        a.data[i] = val;
    }
    return EmlOk;
}


EmlVector
eml_vector_view(EmlVector orig, int start, int end) {
    const int length = end-start;
    const EmlVector view = { orig.data+start, length };
    return view;
}

float
eml_signal_mean(float *data, int length)
{
    float sum = 0.0f;
    for (int32_t i=0; i<length; i++) {
        sum += data[i];
    }
    float mean = sum/length; 
    return mean;
}

// Hann window
EmlError
eml_signal_hann_apply(float *data, int length)
{
    for (int i=0; i<length; i++) {
        float m = (float)(0.5 * (1 - cos(2*M_PI*i/(length-1))));
        data[i] = m * data[i];
    }
    return EmlOk;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EMVECTOR_H
