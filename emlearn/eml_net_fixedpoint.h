
#ifndef EML_NET_FIXEDPOINT_H
#define EML_NET_FIXEDPOINT_H

#include "eml_common.h"
#include "eml_net_common.h"
#include "eml_fixedpoint.h"

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


// Inference for a single layer
EmlError
eml_net_forward_q16(const eml_q16_t *in, int32_t in_length,
                const eml_fixed32_t *weights,
                const eml_fixed32_t *biases,
                EmlNetActivationFunction activation,
                eml_q16_t *out, int32_t out_length)
{

    // multiply inputs by weights
    for (int o=0; o<out_length; o++) {
#if 0
FIXME: implement
        float sum = 0.0f;
        for (int i=0; i<in_length; i++) {
            const int w_idx = o+(i*out_length);
            const float w = weights[w_idx];
            sum += w * in[i];
        }
        out[o] = sum + biases[o];
#endif
    }

    // apply activation function
    if (activation == EmlNetActivationIdentity) {
        // no-op
    } else if (activation == EmlNetActivationRelu) {
        for (int i=0; i<out_length; i++) {
#if 0
FIXME: implement
            out[i] = eml_net_relu(out[i]);
#endif
        }
    } else if (activation == EmlNetActivationLogistic) {
        for (int i=0; i<out_length; i++) {
#if 0
FIXME: implement
            out[i] = eml_net_expit(out[i]);
#endif
        }

    } else if (activation == EmlNetActivationSoftmax) {
#if 0
FIXME: implement
        eml_net_softmax(out, out_length);
#endif
    } else {
        return EmlUnsupported;
    }

    return EmlOk;
}


int32_t
eml_argmax_fixed32(const eml_fixed32_t *values, int values_length)
{
    if (values_length <= 0) {
        return -1;
    }

    eml_fixed32_t vmax = values[0];
    int32_t argmax = -1;
    for (int i=0; i<values_length; i++) {
        if (values[i] > vmax) {
            vmax = values[i];
            argmax = i;
        }
    }
    return argmax;
}


#ifdef __cplusplus
}
#endif
#endif // EML_NET_FIXEDPOINT_H
