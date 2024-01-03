
#ifndef EML_QUANTIZER_H
#define EML_QUANTIZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _EmlQuantizer {
    float scale;
} EmlQuantizer;


EmlError
eml_quantizer_check_forward_int16(EmlQuantizer *self,
    float *values, int values_length,
    int16_t *out, int out_length,
    int *underflows_out,
    int *overflows_out)
{
    EML_PRECONDITION(out_length == values_length, EmlSizeMismatch); 

    int underflows = 0;
    int overflows = 0;
    const float max_value = 32767.0f;
    const float min_value = -32767.0f;

    for (int i=0; i<values_length; i++) {
        float scaled = values[i] * self->scale;
        if (scaled < min_value) {
            underflows += 1;
            scaled = min_value;
        }
        if (scaled > max_value) {
            overflows += 1;
            scaled = max_value;
        }
        out[i] = (int16_t)scaled;
    }

    *underflows_out = underflows;
    *overflows_out = overflows;

    return EmlOk;
}

EmlError
eml_quantizer_forward_int16(EmlQuantizer *self,
    float *values, int values_length,
    int16_t *out, int out_length)
{
    // Preconditions checked in inner function call

    int underflows = -1;
    int overflows = -1;

    const EmlError err = eml_quantizer_check_forward_int16(self, \
            values, values_length, \
            out, out_length, \
            &underflows, &overflows);
    return err;
}


EmlError
eml_quantizer_inverse_int16(EmlQuantizer *self,
    int16_t *values, int values_length,
    float *out, int out_length)
{
    EML_PRECONDITION(out_length == values_length, EmlSizeMismatch); 

    for (int i=0; i<values_length; i++) {
        const float v = (float)values[i]; 
        float scaled = v / self->scale;
        out[i] = scaled;
    }

    return EmlOk;
}


#ifdef __cplusplus
}
#endif

#endif // EML_QUANTIZER_H

