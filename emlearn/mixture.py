
"""
Gaussian Mixture models

Implements the GMM models from scikit-learn,
and reuses some code for compile-time processing.

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_gaussian_mixture.py#L380
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_bayesian_mixture.py
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_base.py

For other GMM implementations in C, see vlfeat

https://www.vlfeat.org/api/gmm.html
https://github.com/vlfeat/vlfeat/blob/master/vl/gmm.c#L712
Note: implements only diagonal covariance matrix
"""


import os.path
import tempfile

import numpy

from . import common, cgen


def get_covariance_type(s):
    if s == 'diag':
        s = 'diagonal'
    return 'EmlCovariance' + s.title()


def generate_code(model, name='fss_mode'):

    means = model._means
    log_det = model._log_det
    covar_type = get_covariance_type(model._covariance_type)
    precisions = model._precisions_col
    log_weights = model._log_weights

    n_components, n_features = means.shape

    means_name = f'{name}_means'
    means_size = n_components * n_features
    means_arr = cgen.array_declare(means_name, size=means_size, values=means.flatten())

    log_dets_name = f'{name}_log_dets'
    log_dets_arr = cgen.array_declare(log_dets_name, values=log_det.flatten())

    precisions_name = f'{name}_precisions'
    precisions_arr = cgen.array_declare(precisions_name, values=precisions.flatten())

    log_weights_name = f'{name}_log_weights'
    log_weights_arr = cgen.array_declare(log_weights_name, values=log_weights.flatten())


    predict_func = f'''
        int32_t
        {name}_log_proba(const float values[], int32_t values_length, float *out)
        {{

            return eml_mixture_log_proba(&{name}_model,
                                values, values_length,
                                out);
        }}


        EmlError
        {name}_score(const float values[], int32_t values_length, float *probabilities, float *score)
        {{

            return eml_mixture_score(&{name}_model,
                                values, values_length,
                                probabilities, score);
        }}
    '''

    model_init = f'EmlMixtureModel {name}_model = ' + cgen.struct_init(
        n_components,
        n_features,
        covar_type,
        means_name,
        precisions_name,
        log_dets_name,
        log_weights_name,
    ) + ';\n'

    preamble = """
    // !! This file was generated using emlearn

    #include <eml_mixture.h>
    """

    out = '\n'.join([
        preamble,
        means_arr,
        precisions_arr,
        log_weights_arr,
        log_dets_arr,
        model_init,
        predict_func,
    ])

    return out


def predict(bin_path, X, verbose=1):
    import subprocess

    def predict_one(x):
        args = [ bin_path ]
        args += [ str(v) for v in x ]
        out = subprocess.check_output(args)
        if verbose > 0:
            print(f"run args={args} out={out} ")

        lines = out.decode('utf-8').split('\n')
        if verbose > 1:
            for line in lines:
                print('l', line)

        outs = lines[-1].split(',')
        values = [ float(s) for s in outs ]
        return values

    y = [ predict_one(x) for x in numpy.array(X) ]
    return numpy.array(y)

def build_executable_proba(wrapper, out_dir, name='gmm'):
    n_components, n_features = wrapper._means.shape

    model_code = generate_code(wrapper, name=name)

    includes = """
    #include <stdio.h> // printf
    #include <stdlib.h> // stdod
    """

    code = includes + model_code + f"""

    static float features[{n_features}] = {{0.0}};
    static float output[{n_components}] = {{0.0}};

    int
    main(int argc, const char *argv[])
    {{
        const int n_features = {n_features};
        const int n_components = {n_components};

        if (argc != 1+n_features) {{
            return -1;
        }}

        for (int i=1; i<argc; i++) {{
            features[i-1] = strtod(argv[i], NULL);
        }}

        const EmlError out = {name}_log_proba(features, n_features, output);
        if (out != EmlOk) {{
            return -out; // error
        }}

        for (int i=0; i<n_components; i++) {{
            printf("%.6f", output[i]);
            if (i != (n_components-1)) {{
                printf(",");
            }}
        }}
        return 0;
    }}
    """
    

    # Compile the xor.c example program
    src_path = os.path.join(out_dir, 'gmm.c')

    with open(src_path, 'w') as f:
        f.write(code)

    include_dirs = [ common.get_include_dir() ]
    bin_path = common.compile_executable(src_path, out_dir, include_dirs=include_dirs)

    return bin_path

def build_executable_score(wrapper, out_dir, name='gmm'):
    n_components, n_features = wrapper._means.shape

    model_code = generate_code(wrapper, name=name)

    includes = """
    #include <stdio.h> // printf
    #include <stdlib.h> // stdod
    """

    code = includes + model_code + f"""

    static float features[{n_features}] = {{0.0}};
    static float output[{n_components}] = {{0.0}};

    int
    main(int argc, const char *argv[])
    {{
        const int n_features = {n_features};
        const int n_components = {n_components};

        if (argc != 1+n_features) {{
            return -1;
        }}

        for (int i=1; i<argc; i++) {{
            features[i-1] = strtod(argv[i], NULL);
        }}

        float score;
        const EmlError out = {name}_score(features, n_features, output, &score);
        if (out != EmlOk) {{
            return -out; // error
        }}

        printf("%f", score);

        return 0;
    }}
    """

    # Compile
    src_path = os.path.join(out_dir, 'gmm.c')

    with open(src_path, 'w') as f:
        f.write(code)

    include_dirs = [ common.get_include_dir() ]
    bin_path = common.compile_executable(src_path, out_dir, include_dirs=include_dirs)

    return bin_path

def convert_to_full(means, precisions_chol, covariance_type):
    """
    Convert the different covariance structures into "full" covariance matrix form.

    This means only a single inference implementation is needed.
    Likely less optimal in terms of run-time, but much simpler.
    """

    n_components, n_features = means.shape

    out = None
    if covariance_type == 'full':
        # already full covariance
        assert len(precisions_chol.shape) == 3, precisions_chol.shape
        out = precisions_chol

    elif covariance_type == 'tied':
        # all components share the same general covariance matrix
        assert len(precisions_chol.shape) == 2, precisions_chol.shape
        assert precisions_chol.shape[0] == n_features
        assert precisions_chol.shape[1] == n_features

        out = [ precisions_chol for _ in range(n_components) ]
        out = numpy.stack(out)

    elif covariance_type == 'diag':
        # each component has its own diagonal covariance matrix
        assert len(precisions_chol.shape) == 2, precisions_chol.shape
        assert precisions_chol.shape[0] == n_components
        assert precisions_chol.shape[1] == n_features

        out = [ numpy.diag(a) for a in precisions_chol ]
        out = numpy.stack(out)

    elif covariance_type == 'spherical':
        # each component has its own single variance
        assert len(precisions_chol.shape) == 1, precisions_chol.shape
        assert precisions_chol.shape[0] == n_components
        # copy out across features, features

        out = [ v * numpy.eye(n_features, n_features) for v in precisions_chol ]
        out = numpy.stack(out)

    else:
        raise ValueError("Unknown covariance_type '{}'")


    assert len(out.shape) == 3, out.shape
    assert out.shape == (n_components, n_features, n_features), out.shape
    return out

def get_log_weights(estimator):
    from scipy.special import digamma

    log_weights = estimator._estimate_log_weights()
    n_components, n_features = estimator.means_.shape

    # For BayesianGaussianMixture   
    # In scikit-learn this is done by adding to log_probs during prediction
    # but we instead bake this into the "weights" 
    # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/mixture/_bayesian_mixture.py#L768
    is_bayesian = hasattr(estimator, 'degrees_of_freedom_') 
    if is_bayesian:
        # Remove `n_features * numpy.log(self.degrees_of_freedom_)` because the precision matrix is normalized
        log_weights = log_weights - (0.5 * n_features * numpy.log(estimator.degrees_of_freedom_))

        log_lambda = n_features * numpy.log(2.0) + numpy.sum(
            digamma(0.5* (estimator.degrees_of_freedom_ - numpy.arange(0, n_features)[:, numpy.newaxis])),
            0,
        )

        log_weights = log_weights + (0.5 * (log_lambda - n_features / estimator.mean_precision_))
        return log_weights

    else:
        return log_weights


class Wrapper:
    def __init__(self, estimator, classifier, dtype='float', verbose=0):
        self.dtype = dtype
        self.verbose = verbose

        n_components, n_features = estimator.means_.shape
        covariance_type = estimator.covariance_type
        precisions_chol = estimator.precisions_cholesky_

        # Convert all types to "full" covariance matrix
        # TODO: native aupport for tied/diag/spherical
        precisions_chol = convert_to_full(estimator.means_, precisions_chol, covariance_type)
        covariance_type = 'full'

        from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky

        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features)

        self._log_det = log_det
        self._means = estimator.means_.copy()
        self._covariance_type = covariance_type
        self._precisions_col = precisions_chol
        self._log_weights = get_log_weights(estimator)


    def predict_proba(self, X):

        with tempfile.TemporaryDirectory() as out_dir:

            bin_path = build_executable_proba(self, out_dir=out_dir)
            c_predictions = predict(bin_path, X, verbose=self.verbose)

        # XXX: note, this is actually log probabilities
        return c_predictions

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = numpy.argmax(probabilities, axis=1)
        return predictions

    def score_samples(self, X):
        from scipy.special import logsumexp

        if True:
            with tempfile.TemporaryDirectory() as out_dir:

                bin_path = build_executable_score(self, out_dir=out_dir)
                predictions = predict(bin_path, X, verbose=self.verbose)

            score = predictions[:,0]

        else:
            prob = self.predict_proba(X)
            score = logsumexp(prob, axis=1)

        return score


    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = generate_code(self, name=name)
        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code

