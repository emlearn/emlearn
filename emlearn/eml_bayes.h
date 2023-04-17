
#ifndef EML_BAYES_H
#define EML_BAYES_H
#ifdef __cplusplus
extern "C" {
#endif

#include "eml_common.h"
#include "eml_fixedpoint.h"

#ifndef EML_MAX_CLASSES
#define EML_MAX_CLASSES 10
#endif

typedef struct _EmlBayesSummary {
    eml_q16_t mean;
    eml_q16_t std;
    eml_q16_t stdlog2;
} EmlBayesSummary;

typedef struct _BayesModel {
   int32_t n_classes;
   int32_t n_features;
   EmlBayesSummary *summaries;
} EmlBayesModel;

// C.S. Turner, "A Fast Binary Logarithm Algorithm"
// https://stackoverflow.com/questions/4657468/fast-fixed-point-pow-log-exp-and-sqrt
static inline
int32_t eml_bayes_log2fix (uint32_t x, size_t precision)
{
    int32_t b = 1U << (precision - 1);
    int32_t y = 0;

    if (precision < 1 || precision > 31) {
        //errno = EINVAL;
        return INT32_MAX; // indicates an error
    }

    if (x == 0) {
        return INT32_MIN; // represents negative infinity
    }

    while (x < 1U << precision) {
        x <<= 1;
        y -= 1U << precision;
    }

    while (x >= 2U << precision) {
        x >>= 1;
        y += 1U << precision;
    }

    uint64_t z = x;

    for (size_t i = 0; i < precision; i++) {
        z = z * z >> precision;
        if (z >= 2U << precision) {
            z >>= 1;
            y += b;
        }
        b >>= 1;
    }

    return y;
}

// log2 of normpdf, implemented using quadratic function
static eml_q16_t
eml_bayes_logpdf_std(eml_q16_t x)
{
    const eml_q16_t a = EML_Q16_FROMFLOAT(-0.7213475204444817);
    const eml_q16_t b = EML_Q16_FROMFLOAT(-1.0005845727355313e-15);
    const eml_q16_t c = EML_Q16_FROMFLOAT(-1.3257480647361592);

    const eml_q16_t axx = eml_q16_mul(a, eml_q16_mul(x, x));
    const eml_q16_t bx = eml_q16_mul(b, x);
    const eml_q16_t y = axx + bx + c;
    return y;
}

// log2 of normal probability density function PDF
// using a scaled/translated standard distribution
static eml_q16_t
eml_bayes_logpdf(eml_q16_t x, eml_q16_t mean, eml_q16_t std, eml_q16_t stdlog2)
{
   const eml_q16_t xm = eml_q16_div((x - mean), std);
   const eml_q16_t xx = (xm > 0) ? xm : -xm;
   const eml_q16_t p = eml_bayes_logpdf_std(xx) - stdlog2;
   return p; 
}


int32_t
eml_bayes_predict(EmlBayesModel *model, const float values[], int32_t values_length) {
   //fprintf(stderr, "predict(%d), classes=%d features=%d f0=%f\n",
   //      values_length, model->n_classes, model->n_features, values[0]);

   EML_PRECONDITION(model, -EmlUninitialized);
   EML_PRECONDITION(values, -EmlUninitialized);
   EML_PRECONDITION(model->n_classes >= 2, -EmlUninitialized);

   eml_q16_t class_probabilities[EML_MAX_CLASSES];

   for (int class_idx = 0; class_idx<model->n_classes; class_idx++) {

      eml_q16_t class_p = 0;
      for (int value_idx = 0; value_idx<values_length; value_idx++) {
         const int32_t summary_idx = class_idx*model->n_features + value_idx;
         EmlBayesSummary summary = model->summaries[summary_idx];
         const eml_q16_t val = EML_Q16_FROMFLOAT(values[value_idx]);
         const eml_q16_t plog = eml_bayes_logpdf(val, summary.mean, summary.std, summary.stdlog2);

         class_p += plog;
      }
      class_probabilities[class_idx] = class_p;
      //printf("class %d : %f\n", class_idx, p);
   }

   eml_q16_t highest_prob = class_probabilities[0];
   int32_t highest_idx = 0;
   for (int class_idx = 1; class_idx<model->n_classes; class_idx++) {
      const eml_q16_t p = class_probabilities[class_idx]; 
      if (p > highest_prob) {
         highest_prob = p;
         highest_idx = class_idx;
      }
   }
   return highest_idx;
}


#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_BAYES_H
