
#ifndef EML_FIXEDPOINT_H
#define EML_FIXEDPOINT_H

#ifdef __cplusplus
extern "C" {
#endif

// Fixed-point helpers
typedef int32_t eml_q16_t;
#define EML_Q16_FRACT_BITS 16
#define EML_Q16_ONE (1 << EML_Q16_FRACT_BITS)
#define EML_Q16_FROMINT(x) ((x) << EML_Q16_FRACT_BITS)
#define EML_Q16_FROMFLOAT(x) ((int)((x) * (1 << EML_Q16_FRACT_BITS))) 
#define EML_Q16_TOINT(x) ((x) >> EML_Q16_FRACT_BITS)
#define EML_Q16_TOFLOAT(x) (((float)(x)) / (1 << EML_Q16_FRACT_BITS))

// TODO: namespace properly

// Fixed-point math
#define eml_q16_mul(x, y) ( ((x) >> EML_Q16_FRACT_BITS/2) * ((y)>> EML_Q16_FRACT_BITS/2) )

static eml_q16_t
eml_q16_div(eml_q16_t a, eml_q16_t b)
{
    int64_t temp = (int64_t)a << EML_Q16_FRACT_BITS;
    if((temp >= 0 && b >= 0) || (temp < 0 && b < 0)) {   
        temp += b / 2;
    } else {
        temp -= b / 2;
    }
    return (int32_t)(temp / b);
}

#ifdef __cplusplus
}
#endif

#endif // EML_FIXEDPOINT_H
