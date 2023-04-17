
#ifndef EML_MIXTURE_H
#define EML_MIXTURE_H

#include <math.h>

#include "eml_common.h"
#include "eml_fixedpoint.h"

#ifdef __cplusplus
extern "C" {
#endif

// numpy.log(2*numpy.pi)
#define EML_LOG_2PI 1.8378770664093453

bool
eml_dot_product(float *a, float *b, int n)
{
    float sum = 0; 
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
 
    return sum;
}

EmlError
eml_minmax(const float *arr, int length, float *out_min, float *out_max)
{
    float min = +INFINITY;
    float max = -INFINITY;

    for (int i=0; i<length; i++) {
        const float value = arr[i];

        if (value > max) {
            max = value;
        }
        if (value < min) {
            min = value;
        }
    }

    if (out_min) {
        *out_min = min;
    }
    if (out_max) {
        *out_max = max;
    }

    return EmlOk;
}



EmlError
eml_logsumexp(const float *arr, int length, float *out_sum)
{
    float a_max = 0.0f;

    // For numerical stability, scale down the numbers before exp
    EmlError err = eml_minmax(arr, length, NULL, &a_max);
    if (err != EmlOk) {
        return err;
    }

    float sum = 0.0f;
    for (int i=0; i<length; i++) {
        const float tmp = expf(arr[i] - a_max);
        sum += tmp;
    }

    const float out = logf(sum) + a_max;
    *out_sum = out;

    return EmlOk;
}




typedef enum EmlCovarianceType_ {
    EmlCovarianceFull,
    EmlCovarianceTied,
    EmlCovarianceDiagonal,
    EmlCovarianceSpherical,
} EmlCovarianceType;

typedef struct _EmlMixtureModel {
   int32_t n_components;
   int32_t n_features;
   EmlCovarianceType covariance_type;

   const float *means; // n_components * n_features (in the full covariance case)

   // Cholesky decompositions of the precision matrices.
   // Length depends on covariance_type, see eml_mixture_precisions_length
   const float *precisions;

    // FIXME: combine log_dets and log_weights?
   const float *log_dets; // n_components
   const float *log_weights; // n_components
} EmlMixtureModel;


int
eml_mixture_precisions_length(EmlMixtureModel *model)
{
    int length = -1;
    const int32_t features = model->n_features;
    const int32_t components = model->n_components;

    switch (model->covariance_type) {
        case EmlCovarianceFull:
            length = (components * features * features);
            break;
        case EmlCovarianceTied:
            length = (features * features);
            break;
        case EmlCovarianceDiagonal:
            length = (components * features);
            break;
        case EmlCovarianceSpherical:
            length = (components);
            break;
    }

    return length;
}

#if 0
void
print_array(const float *array, int n) {
    printf("[");
    for (int i=0; i<n; i++) {
        printf("%.4f ", array[i]);
    }
    printf("]");
}
#endif

int32_t
eml_mixture_log_proba(EmlMixtureModel *model,
                        const float values[], int32_t values_length,
                        float *probabilities)
{

    EML_PRECONDITION(model, -EmlUninitialized);
    EML_PRECONDITION(values, -EmlUninitialized);
    EML_PRECONDITION(model->n_components > 0, -EmlUninitialized);

    float *out = probabilities;
    const int n_features = model->n_features;

    for (int c=0; c<model->n_components; c++) {
        
        const float *means = model->means + (c*n_features);
      
#if 0  
        printf("means: ");
        print_array(means, n_features);
        printf("\n");
#endif

        float log_prob = 0.0;

        switch (model->covariance_type) {

            case EmlCovarianceFull:

                for (int f=0; f<n_features; f++) {

                    const float *precisions = model->precisions + (c*n_features*n_features);
                    const int n_precisions = model->n_features; // per feature, for this component 
#if 0
                    printf("precisions: ");
                    print_array(precisions, n_precisions);
                    printf("\n");
#endif
                    float dot_x = 0.0;
                    float dot_m = 0.0;
                    for (int p=0; p<n_precisions; p++) {
                        dot_x += (values[p] * precisions[(p*n_features)+f]);
                        dot_m += (means[p] * precisions[(p*n_features)+f]);
                    }
                    const float y = (dot_x - dot_m);
#if 0
                    printf("c_yy component=%d feature=%d y=%.4f dot_x=%.4f dot_m=%.4f  \n",
                            c, f, y, dot_x, dot_m);
#endif

                    log_prob += (y*y);
                }
#if 0
                printf("c_log_prob component=%d log_prob=%.4f  \n", c, log_prob);
#endif

                break;

            case EmlCovarianceTied:
                return EmlUnsupported;
                break;

            case EmlCovarianceDiagonal:
                return EmlUnsupported;
                break;

            case EmlCovarianceSpherical:
                return EmlUnsupported;
                break;

        }

        out[c] = -0.5 * (n_features * EML_LOG_2PI + log_prob ) + model->log_dets[c];

#if 0
        printf("c_s component=%d s=%.4f log_det=%.4f weight=%.4f \n",
                c, out[c], model->log_dets[c], model->log_weights[c]);
#endif

        out[c] += model->log_weights[c];

    }


   return EmlOk;
}

int32_t
eml_mixture_score(EmlMixtureModel *model,
                    const float values[], int32_t values_length,
                    float *probabilities,
                    float *out_score)
{

    EML_PRECONDITION(model, -EmlUninitialized);
    EML_PRECONDITION(values, -EmlUninitialized);
    EML_PRECONDITION(model->n_components > 0, -EmlUninitialized);


    EmlError status = \
        eml_mixture_log_proba(model, values, values_length, probabilities);
    if (status != EmlOk) {
        return status;
    }

    float score;
    status = eml_logsumexp(probabilities, model->n_components, &score);
    if (status != EmlOk) {
        return status;
    }

    *out_score = score;

    return EmlOk;
}

#ifdef __cplusplus
}
#endif

#endif // EML_MIXTURE_H
