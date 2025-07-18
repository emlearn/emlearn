
"""
Gaussian Naive Bayes
=========================
"""

import numpy
numpy.seterr(all='raise')

from . import common, cgen

import os.path

def prob_ref(x, mean, std):
    exponent = (- ((x - mean)**2 / (2 * std**2)))
    sigma_max = 100
    if exponent < -sigma_max:
        exponent = -sigma_max
    return (numpy.exp(exponent) / (numpy.sqrt(2 * numpy.pi) * std))

def c_struct_init(vals, convert):
    if convert is None:
      convert = str
    s = ','.join(convert(v) for v in vals)
    return '{ ' + s + ' }'

def c_tofixed(v):
    return "EML_Q16_FROMFLOAT({})".format(v)


def generate_c(model, name='myclassifier'):
    n_classes, n_features, n_attributes = model.shape
    assert n_attributes == 3 # mean+std+stdlog2 

    cgen.assert_valid_identifier(name)

    summaries_data = []
    for class_n, class_summaries in enumerate(model):
        for feature_n, summary in enumerate(class_summaries):
            summaries_data.append(list(summary))

    summaries_name = name + '_summaries'
    summaries = """EmlBayesSummary {name}[{items}] = {{
        {summaries_init}
    }};
    """.format(**{
        'name': summaries_name,
        'items': n_classes*n_features,
        'summaries_init': ',\n  '.join(c_struct_init(d, c_tofixed) for d in summaries_data)
    })

    model = """EmlBayesModel {name} = {{
        {classes},
        {features},
        {summaries},
    }};
    """.format(**{
        'name': name+'_model',
        'classes': n_classes,
        'features': n_features,
        'summaries': summaries_name,
    })

    head = """// !!! This file is generated by emlearn !!!

    #include <eml_bayes.h>
    """

    predict_function = f"""
    int32_t
    {name}_predict(const float *features, int32_t n_features)
    {{
        return eml_bayes_predict(&{name}_model, features, n_features);

    }}
    """

    return '\n\n'.join([head, summaries, model, predict_function]) 


# TODO: support class_priors_
# TODO: support predict_log_proba / predict_proba
class Wrapper(object):
    """
    Python API for Bayes classifier implemented in C
    """
    def __init__(self, estimator, method):

        # FIXME: use var,mean numpy arrays directly
        n_classes, n_features = estimator.theta_.shape
        model = numpy.ndarray(shape=(n_classes, n_features, 3), dtype=float)

        if hasattr(estimator, 'var_'):
            # 1.0 and later
            variance = estimator.var_
        elif hasattr(estimator, 'sigma_'):
            # pre 1.0
            variance = estimator.sigma_


        for class_n in range(0, n_classes):
            for feature_n in range(0, n_features):
                mean = estimator.theta_[class_n,feature_n]
                std = numpy.sqrt(variance[class_n,feature_n])
                std_log2 = numpy.log2(std)
                model[class_n,feature_n] = (mean, std, std_log2)
        self.model = model

        if method is None:
            method = 'loadable'

        if method == 'loadable':
            name = 'mybayes'
            func = 'eml_bayes_predict(&{}_model, values, length)'.format(name)
            code = self.save(name=name)
            self.classifier = common.CompiledClassifier(code, name=name, call=func)
        elif method == 'inline':
            raise NotImplementedError('NaiveBayes does not support inline C code generation')
        else:
            raise ValueError(f"Unsupported inference method '{method}'")


    def predict(self, X):
        return self.classifier.predict(X)

    def save(self, file=None, name=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = generate_c(self.model, name)
        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code
