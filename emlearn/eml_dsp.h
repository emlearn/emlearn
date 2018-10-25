
#ifndef EML_DSP_H
#define EML_DSP_H

#include "eml_common.h"


typedef enum _EmlConvolveMode {
    EmlConvolveFull = 0,
    EmlConvolveSame,
    EmlConvolveValid,
    EmlConvolveModes,
} EmlConvolveMode;

static const char *
eml_convolve_mode_strs[EmlConvolveModes] = {
    "full",
    "same",
    "valid",
};

bool
eml_convolve_mode_valid(EmlConvolveMode e) {
    return (e >= 0 && e < EmlConvolveModes);
}

const char *
eml_convolve_mode_str(EmlConvolveMode v) {
    if (eml_convolve_mode_valid(v)) {
        return eml_convolve_mode_strs[v];
    } else {
        return NULL;
    }
}

int32_t
eml_convolve_1d_length(int32_t in_length, int8_t kernel_length, EmlConvolveMode mode)
{
    if (mode == EmlConvolveFull) {
        return in_length+kernel_length-1;
    } else if (mode == EmlConvolveSame) {
        return eml_max_int(in_length, kernel_length);
    } else if (mode == EmlConvolveValid) {
        return eml_max_int(in_length, kernel_length) - eml_min_int(in_length, kernel_length) + 1;
    } else {
        return 0;
    }
}

// Internal function
// Computes the full convolution. Extends output at the end with boundary results
EmlError
eml_convolve_1d_full(const float *in, int32_t in_length,
                float *out, int32_t out_length,
                const float *kernel, int8_t kernel_size,
                int8_t stride)
{
    const int32_t out_max = eml_convolve_1d_length(in_length, kernel_size, EmlConvolveFull);
    EML_PRECONDITION(out_max > 0, EmlUnknownError);
    EML_PRECONDITION(out_length >= out_max, EmlSizeMismatch);

    for (int32_t n = 0; n < out_max; n+=stride) {
        const int32_t kmin = (n >= kernel_size-1) ? n - (kernel_size-1) : 0;
        const int32_t kmax = eml_min_int(n, in_length-1);
        out[n] = 0;
        for (int32_t k = kmin; k <= kmax; k++) {
            out[n] += in[k] * kernel[n - k];
        }
    }
    return EmlOk;
}

EmlError
eml_convolve_1d_valid(const float *in, int32_t in_length,
                float *out, int32_t out_length,
                const float *kernel, int8_t kernel_size,
                int8_t stride)
{
    const int32_t out_max = eml_convolve_1d_length(in_length, kernel_size, EmlConvolveValid);
    EML_PRECONDITION(out_max > 0, EmlUnknownError);
    EML_PRECONDITION(out_length >= out_max, EmlSizeMismatch);

    for (int32_t i=0; i < out_max; ++i) {
        out[i] = 0;
        for (int32_t j=kernel_size-1, k=i; j >= 0; --j) {
            out[i] += kernel[j] * in[k];
            ++k;
        }
    }
    return EmlOk;
}



// Output same size as input, result centered, with zero-padded out-of-bounds 
EmlError
eml_convolve_1d_same(const float *in, int32_t in_length,
                float *out, int32_t out_length,
                const float *kernel, int8_t kernel_size,
                int8_t stride)
{
    const int32_t out_max = eml_convolve_1d_length(in_length, kernel_size, EmlConvolveSame);
    EML_PRECONDITION(out_max > 0, EmlUnknownError);
    EML_PRECONDITION(out_length >= out_max, EmlSizeMismatch);

    const int8_t kernel_half = kernel_size/2;

    for (int32_t n = 0; n < out_max; n+=stride) {

        const int32_t kmin = (n >= kernel_size-1) ? n - (kernel_size-1) : 0;
        const int32_t kmax = eml_min_int(kmin+kernel_size-1, in_length-1);

        fprintf(stderr, "n=%d, kmin=%d, kmax=%d, kdiff=%d\n",
                        n, kmin, kmax, kmax-kmin);

        out[n] = 0;
        for (int32_t k = kmin; k <= kmax; k++) {
            out[n] += in[k+kernel_half] * kernel[n - k];
        }
    }
    return EmlOk;
}

EmlError
eml_convolve_1d(const float *in, int32_t in_length,
                float *out, int32_t out_length,
                const float *kernel, int8_t kernel_size,
                int8_t stride, EmlConvolveMode mode)
{
    EML_PRECONDITION(in, EmlUninitialized);
    EML_PRECONDITION(out, EmlUninitialized);
    EML_PRECONDITION(kernel, EmlUninitialized);
    EML_PRECONDITION(in_length > 0, EmlUninitialized);
    EML_PRECONDITION(out_length > 0, EmlUninitialized);
    EML_PRECONDITION(stride > 0, EmlInvalidParameter);
    EML_PRECONDITION(kernel_size > 0, EmlInvalidParameter);
    EML_PRECONDITION(eml_convolve_mode_valid(mode), EmlInvalidParameter);

    if (mode == EmlConvolveFull) {
        EML_CHECK_ERROR(eml_convolve_1d_full(in, in_length, out, out_length,
                                             kernel, kernel_size, stride));
    } else if (mode == EmlConvolveSame) {
        EML_CHECK_ERROR(eml_convolve_1d_same(in, in_length, out, out_length,
                                             kernel, kernel_size, stride));
    } else if (mode == EmlConvolveValid) {
        EML_CHECK_ERROR(eml_convolve_1d_valid(in, in_length, out, out_length,
                                             kernel, kernel_size, stride));
    } else {
        EML_POSTCONDITION(false, EmlUnreachable);
    }

    return EmlOk;
}

// MAYBE: support special-case for 3x3 kernel, using Straessen?

#endif
