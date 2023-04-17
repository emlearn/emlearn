
#ifndef EML_IIR
#define EML_IIR

#include "eml_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Each stage is implemented using a Biquad filter
// https://en.wikipedia.org/wiki/Digital_biquad_filter
// TransporedDirectForm2
// preferred for floating-point
float
eml_biquad_tdf2(float s1[2], float s2[2],
                const float a[3], const float b[3],
                float in)
{
    const float out = s1[1] + b[0]*in;
    s1[0] = s2[1] + b[1]*in - a[1]*out;
    s2[0] = b[2]*in - a[2]*out;
    s1[1] = s1[0];
    s2[1] = s2[0];
    return out;
}

// TODO: implement DirectForm1 for fixed-point.
// Should possibly also use first-order noise shaping


// IIR filters using cascades of Second Order Sections (SOS)
// Follows conventions of scipy.signal.sosfilt
//
// A single second-order filter is just a special case with n_stages=1
typedef struct _EmlIIR {
    int n_stages;
    float *states;
    int states_length;// n_stages * 4
    const float *coefficients; // In scipy.signal.sosfilt convention. [0..2]: numerator, [3..5]: denominator
    int coefficients_length; // n_stages * 6.
} EmlIIR;


EmlError
eml_iir_check(EmlIIR filter) {
    EML_PRECONDITION(filter.n_stages >= 1, EmlUninitialized);
    EML_PRECONDITION(filter.states, EmlUninitialized);
    EML_PRECONDITION(filter.coefficients, EmlUninitialized);
    EML_PRECONDITION(filter.states_length == filter.n_stages*4, EmlSizeMismatch);
    EML_PRECONDITION(filter.coefficients_length == filter.n_stages*6, EmlSizeMismatch);
    return EmlOk;
}

float
eml_iir_filter(EmlIIR filter, float in) {

    float out = in;
    for (int stage=0; stage < filter.n_stages; stage++) {
        const float *num = filter.coefficients + ((6*stage)+0);
        const float *den = filter.coefficients + ((6*stage)+3);
        float *s1 = filter.states + ((4*stage)+0);
        float *s2 = filter.states + ((4*stage)+2);
        out = eml_biquad_tdf2(s1, s2, den, num, out);
    }
    return out;
}

#ifdef __cplusplus
}
#endif

#endif // EML_IIR
