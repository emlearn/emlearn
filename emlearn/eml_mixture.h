

#ifndef EML_MIXTURE_H
#define EML_MIXTURE_H

#include "eml_common.h"
#include "eml_fixedpoint.h"

#ifndef EML_MAX_CLASSES
#define EML_MAX_CLASSES 10
#endif


// numpy.log2(2*numpy.pi)
#define EML_LOG2_2PI 2.651496129472319f

bool
eml_dot_product(float *a, float *b, int n)
{
    float sum = 0; 
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
 
    return sum;
}

typedef enum EmlCovarianceType_ = {
    EmlCovarianceFull,
    EmlCovarianceTied,
    EmlCovarianceDiagonal,
    EmlCovarianceSpherical,
} EmlCovarianceType;

typedef struct _EmlMixtureModel {
   int32_t n_components;
   int32_t n_features;
   EmlCovarianceType covariance_type;

   float *means; // n_components * n_features (in the full covariance case)

   // Cholesky decompositions of the precision matrices.
   // Length depends on covariance_type, see eml_mixture_precisions_length
   float *precisions;

    // FIXME: combine log_dets and log_weights?
   float *log_dets; // n_components
   float *log_weigths; // n_components
} EmlMixtureModel;


int
eml_mixture_precisions_length(EmlMixtureModel *model)
{
    int length = -1;
    const int32_t features = model->n_features;
    const int32_t components = model->n_components;
    const int32_t features = model->n_features;

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



int32_t
eml_mixture_predict_proba(EmlMixtureModel *model,
                        const float values[], int32_t values_length,
                        float *probabilities)
{

   EML_PRECONDITION(model, -EmlUninitialized);
   EML_PRECONDITION(values, -EmlUninitialized);
   EML_PRECONDITION(model->n_components > 0, -EmlUninitialized);


   float *out = probabilities;

    for (int c=0; c<n_components; c++) {

        
        const float *means = model->means[(c*n_features)]

        switch (covariance_type) {

            case EmlCovarianceFull:

                const float *precisions = model->precisions[(c*n_features)]
                const int n_precisions = model->n_features; // per feature, for this component 

                float log_prob = 0.0;
                for (int f=0; f<n_features; f++) {

                    float dot_x = 0.0;
                    float dot_m = 0.0;
                    for (int p=0; p<n_precisions; p++) {
                        dot_x += (values[f] * precisions[p]);
                        dot_m += (means[f] * precisions[p]);
                    }
                    const float y = (dot_x - dot_m);

                    log_prob += (y*y);
                }

                break;

            case EmlCovarianceTied:
                return -66;
                break;

            case EmlCovarianceDiagonal:
                return -66;
                break;

            case EmlCovarianceSpherical:
                return -66;
                break;

        }

        out[c] = -0.5 * (n_features * EML_LOG2_2PI + log_prob ) + log_det[c];
        out[c] += log_weights[c];

    }


*/


   return ;
}



#endif // EML_MIXTURE_H
