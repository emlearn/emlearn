
"""
XOR
===========================

Work in progress
"""

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
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))

    rng = np.random.RandomState(seed)
    X = rng.randn(300, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

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
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)


model = train_model(dataset)


# %%
# Convert model to C using emlearn
# ------------------------
#
# 
def convert_model():

    model_filename = 'digits.h'
    cmodel = emlearn.convert(model)
    code = cmodel.save(file=filename)


# %%
# Use C model to make predictions
# ------------------------
#
# See xor.c
def predict(bin_path, X):

    def predict_one(x):
        # C program takes features A and B as commandline arguments
        # writes the predicted class to stdout
        args = [ bin_path, str(x[0]), str(x[1]) ]
        out = submodule.check_output(args)
        cls = int(out)
        return cls

    y = [ predict_one(x) for x in X ]
    return pandas.Series(y)

def c_model_predict(dataset)
    src_path = os.path.join(here, 'xorg.c')
    bin_path = compile_executable(src_path)

    sub

    pass


