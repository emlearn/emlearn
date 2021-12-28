
"""
XOR
===========================

Work in progress
"""

import os.path

import emlearn
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# %%
# Create dataset
# ------------------------
#
# 
def make_noisy_xor(seed=42):
    xx, yy = numpy.meshgrid(numpy.linspace(-3, 3, 500),
                         numpy.linspace(-3, 3, 500))

    rng = numpy.random.RandomState(seed)
    X = rng.randn(300, 2)
    y = numpy.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # XOR is too easy for factorization machines, so add noise :)
    flip = rng.randint(300, size=15)
    y[flip] = ~y[flip]

    return X, y

dataset = make_noisy_xor()


# %%
# Train ML model
# ------------------------
#
# Usin the standard process with scikit-learn
def train_model(dataset, seed=42):

    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)

    #ypred = model.predict(X_test)

    return model


model = train_model(dataset)


# %%
# Convert model to C using emlearn
# ------------------------
#
# 
def convert_model(model):

    model_filename = os.path.join(os.path.dirname(__file__), 'xor_model.h')
    cmodel = emlearn.convert(model)
    code = cmodel.save(file=model_filename, name='xor')

    assert os.path.exists(model_filename)
    print(f"Generated {model_filename}")

convert_model(model)

# %%
# Use C model to make predictions
# ------------------------
#
# See xor.c
#
# .. literalinclude:: ../../examples/xor.c
#  :language: C
def predict(bin_path, X):
    import subprocess

    def predict_one(x):
        # C program takes features A and B as commandline arguments
        # writes the predicted class to stdout
        args = [ bin_path, str(x[0]), str(x[1]) ]
        out = subprocess.check_output(args)
        cls = int(out)
        return cls

    y = [ predict_one(x) for x in X ]
    return numpy.array(y)

def c_model_predict(dataset):
    out_dir = './examples'
    src_path = os.path.join(os.path.dirname(__file__), 'xor.c')
    include_dirs = [ emlearn.includedir ]
    bin_path = emlearn.common.compile_executable(src_path, out_dir, include_dirs=include_dirs)

    print('Binary', bin_path)
    X, y = dataset
    y_pred_c = predict(bin_path, X)
    y_pred_py = model.predict(X)

    import sklearn.metrics
    f1_score_c = sklearn.metrics.f1_score(y, y_pred_c)
    f1_score_py = sklearn.metrics.f1_score(y, y_pred_py)

    print('F1', f1_score_py, f1_score_c)

    pass

c_model_predict(dataset)

