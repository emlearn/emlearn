
#ifndef EML_BAYES_H
#define EML_BAYES_H

#include <stdint.h>
#include <stddef.h>


// Fixed-point helpers
typedef int32_t val_t;
#define VAL_FRACT_BITS 16
#define VAL_ONE (1 << VAL_FRACT_BITS)
#define VAL_FROMINT(x) ((x) << VAL_FRACT_BITS)
#define VAL_FROMFLOAT(x) ((int)((x) * (1 << VAL_FRACT_BITS))) 
#define VAL_TOINT(x) ((x) >> VAL_FRACT_BITS)
#define VAL_TOFLOAT(x) (((float)(x)) / (1 << VAL_FRACT_BITS))

// TODO: namespace properly

// Fixed-point math
#define val_mul(x, y) ( ((x) >> VAL_FRACT_BITS/2) * ((y)>> VAL_FRACT_BITS/2) )

static val_t
val_div(val_t a, val_t b)
{
    int64_t temp = (int64_t)a << VAL_FRACT_BITS;
    if((temp >= 0 && b >= 0) || (temp < 0 && b < 0)) {   
        temp += b / 2;
    } else {
        temp -= b / 2;
    }
    return (int32_t)(temp / b);
}


typedef struct _EmlBayesSummary {
    val_t mean;
    val_t std;
    val_t stdlog2;
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
static val_t
eml_bayes_logpdf_std(val_t x)
{
    const val_t a = VAL_FROMFLOAT(-0.7213475204444817);
    const val_t b = VAL_FROMFLOAT(-1.0005845727355313e-15);
    const val_t c = VAL_FROMFLOAT(-1.3257480647361592);

    const val_t axx = val_mul(a, val_mul(x, x));
    const val_t bx = val_mul(b, x);
    const val_t y = axx + bx + c;
    return y;
}

// log2 of normal probability density function PDF
// using a scaled/translated standard distribution
static val_t
eml_bayes_logpdf(val_t x, val_t mean, val_t std, val_t stdlog2)
{
   const val_t xm = val_div((x - mean), std);
   const val_t xx = (xm > 0) ? xm : -xm;
   const val_t p = eml_bayes_logpdf_std(xx) - stdlog2;
   return p; 
}


int32_t
eml_bayes_predict(EmlBayesModel *model, const val_t values[], int32_t values_length) {
   //printf("predict(%d), classes=%d features=%d\n",
   //      values_length, model->n_classes, model->n_features);

   const int MAX_CLASSES = 10;
   val_t class_probabilities[MAX_CLASSES];

   for (int class_idx = 0; class_idx<model->n_classes; class_idx++) {

      val_t class_p = 0;
      for (int value_idx = 0; value_idx<values_length; value_idx++) {
         const int32_t summary_idx = class_idx*model->n_features + value_idx;
         EmlBayesSummary summary = model->summaries[summary_idx];
         const val_t val = values[value_idx];
         const val_t plog = eml_bayes_logpdf(val, summary.mean, summary.std, summary.stdlog2);

         class_p += plog;

      }
      class_probabilities[class_idx] = class_p;
      //printf("class %d : %f\n", class_idx, p);
   }

   val_t highest_prob = class_probabilities[0];
   int32_t highest_idx = 0;
   for (int class_idx = 1; class_idx<model->n_classes; class_idx++) {
      const val_t p = class_probabilities[class_idx]; 
      if (p > highest_prob) {
         highest_prob = p;
         highest_idx = class_idx;
      }
   }
   return highest_idx;
}



#endif // EML_BAYES_H
