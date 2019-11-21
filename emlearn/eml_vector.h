
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
eml_vector_shift(EmlVector a, int amount)
{
    if (abs(amount) >= a.length) {
        return -2; // non-sensical
    }

    if (amount == 0) {
        return 0;
    } else if (amount > 0) {
        return -1; // TODO: implement
    } else {
        for (int i=a.length+amount; i<a.length; i++) {  
            a.data[i+amount] = a.data[i];
        }
        return 0;
    }
}

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

#define EM_MAX(a, b) (a > b) ? a : b

int
eml_vector_max_into(EmlVector a, EmlVector b) {
    if (a.length != b.length) {
        return -1;
    }

    for (int32_t i=0; i<a.length; i++) {
        a.data[i] = EM_MAX(a.data[i], b.data[i]);
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

float
eml_vector_mean(EmlVector v) {
    float sum = 0.0f;
    for (int32_t i=0; i<v.length; i++) {
        sum += v.data[i];
    }
    float mean = sum/v.length; 
    return mean;
}

int
eml_vector_subtract_value(EmlVector v, float val) {

    for (int32_t i=0; i<v.length; i++) {
        v.data[i] -= val;
    }
    return 0;
}

EmlVector
eml_vector_view(EmlVector orig, int start, int end) {
    const int length = end-start;
    const EmlVector view = { orig.data+start, length };
    return view;
}

// Mean subtract normalization
int
eml_vector_meansub(EmlVector inout) {

    const float mean = eml_vector_mean(inout);
    eml_vector_subtract_value(inout, mean);

    return 0;
}


// Hann window
EmlError
eml_vector_hann_apply(EmlVector out) {

    const long len = out.length;
    for (int i=0; i<len; i++) {
        float m = (float)(0.5 * (1 - cos(2*M_PI*i/(len-1))));
        out.data[i] = m * out.data[i];
    }
    return EmlOk;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EMVECTOR_H
