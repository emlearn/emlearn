
#ifndef EML_DISTANCE_H
#define EML_DISTANCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <eml_common.h>

/*
Mahalanobis distance

computes (x1 - x2).T * VI * (x1 - x2)
where VI is the precision matrix, the inverse of the covariance matrix
*/
float
eml_mahalanobis_distance_squared(const float *x1, const float *x2,
                const float *precision, int n_features)
{
    float distance = 0.0;

    for (int i=0; i<n_features; i++) {
        float accumulate = 0.0f;
        for (int j=0; j<n_features; j++) {
            accumulate += (precision[j*n_features+i] * (x1[j] - x2[j]));
        }
        distance += (accumulate * (x1[i] - x2[i]));
    }

    return distance;
}

typedef struct _EmlEllipticEnvelope {
    int n_features;
    float decision_boundary;
    const float *means; // shape: n_features
    const float *precision; // shape: n_features*n_features
} EmlEllipticEnvelope;

/* If distance is negative, is outlier */
float
eml_elliptic_envelope_predict(const EmlEllipticEnvelope *self,
                            const float *features, int n_features,
                            float *out_dist)
{
    EML_PRECONDITION(n_features == self->n_features, EmlSizeMismatch);

    const float dist = \
        eml_mahalanobis_distance_squared(features, self->means, self->precision, n_features);
    const float dist_from_boundary = (-dist) - self->decision_boundary;

    const int outlier = (dist_from_boundary < 0 ) ? 1 : 0;

    if (out_dist) {
        //*out_dist = dist_from_boundary;
        *out_dist = dist;
    }

    return outlier;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_FFT_H
