
import emlearn
import eml_net

import pytest
import sklearn
import numpy
from numpy.testing import assert_equal, assert_almost_equal

import sys
import warnings
warnings.filterwarnings(action='ignore', category=sklearn.exceptions.ConvergenceWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

def test_unsupported_activation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=10)
        model.fit([[1.0]], [True])
    with pytest.raises(Exception) as ex:
        model.activation = 'fake22'
        emlearn.convert(model)
    assert 'Unsupported activation' in str(ex.value)
    assert 'fake22' in str(ex.value)


SKLEARN_PARAMS = [
    ( dict(hidden_layer_sizes=(4,), activation='relu'), {'classes': 3, 'features': 2}),
    ( dict(hidden_layer_sizes=(4,), activation='tanh'), {'classes': 2, 'features': 3}),
    ( dict(hidden_layer_sizes=(4,5,3)), {'classes': 5, 'features': 5}),
]

@pytest.mark.parametrize('modelparams,params', SKLEARN_PARAMS)
def test_sklearn_predict(modelparams,params):

    model = MLPClassifier(**modelparams, max_iter=10)

    for random in range(0, 3):
        # create dataset
        rng = numpy.random.RandomState(0)
        X, y = make_classification(n_features=params['features'], n_classes=params['classes'],
                                   n_redundant=0, n_informative=params['features'],
                                   random_state=rng, n_clusters_per_class=1, n_samples=50)
        X += 2 * rng.uniform(size=X.shape)
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

            cmodel = emlearn.convert(model)

            X_test = X_test[:3]
            cproba = cmodel.predict_proba(X_test)
            proba = model.predict_proba(X_test)
            cpred = cmodel.predict(X_test)
            pred = model.predict(X_test)

        assert_almost_equal(proba, cproba, decimal=6)
        assert_equal(pred, cpred)


def keras_mlp_multiclass_activation_layers(features, classes, activation='relu'):
    model = Sequential([
        Dense(8, input_dim=features),
        Activation(activation),
        Dense(classes),
        Activation('softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, dict(features=features, classes=classes) 

def keras_mlp_binary_activation_params(features, activation='relu'):
    model = Sequential([
        Dense(16, input_dim=features, activation=activation),
        Dense(8, activation=activation),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model, dict(features=features, classes=2)

def keras_dropout_relu_softmax(features, classes):
    # Using the specific layer names instead of general Dense,Activation
    model = Sequential([
        Dense(8, input_shape=(features,)),
        keras.layers.ReLU(),
        keras.layers.Dropout(rate=0.5),
        Dense(classes),
        keras.layers.Softmax(),
    ])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, dict(features=features, classes=classes)

# TODO: support CNNs. Conv1D/2D, (ZeroPadding1D/2D), Average/MaxPooling1D/2D, Flatten
# TODO: support simple functional Models, like Logistic Regression. Input+Dense+Softmax

KERAS_MODELS = {
    'MLP.binary': keras_mlp_binary_activation_params(3),
    'MLP.4ary.actlayer': keras_mlp_multiclass_activation_layers(3, 4),
}

if getattr(keras.layers, 'ReLu', None):
    KERAS_MODELS['Dropout.Relu.Softmax'] = keras_dropout_relu_softmax(3, 4),

def assert_equivalent(model, X_test, n_classes, method):
    cmodel = emlearn.convert(model, method=method)

    # TODO: support predict_proba, use that instead
    cpred = cmodel.predict(X_test)
    pred = model.predict(X_test)
    if n_classes == 2:
        pred = (pred[:,0] > 0.5).astype(int)
    else:
        pred = numpy.argmax(pred, axis=1)

    assert_equal(pred, cpred)

@pytest.mark.parametrize('modelname', KERAS_MODELS.keys())
def test_net_keras_predict(modelname):
    model, params = KERAS_MODELS[modelname]

    for random in range(0, 3):
        # create dataset
        rng = numpy.random.RandomState(0)
        X, y = make_classification(n_features=params['features'], n_classes=params['classes'],
                                   n_redundant=0, n_informative=params['features'],
                                   random_state=rng, n_clusters_per_class=1, n_samples=50)
        X += 2 * rng.uniform(size=X.shape)
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        if params['classes'] != 2:
            class_names = None
            y_train = MultiLabelBinarizer(classes=class_names).fit_transform(y_train.reshape(-1, 1))

        model.fit(X_train, y_train, epochs=1, batch_size=10)
        X_test = X_test[:3]

        # check each method. Done here instead of using parameters to save time, above is slow
        assert_equivalent(model, X_test[:3], params['classes'], method='pymodule')
        assert_equivalent(model, X_test[:3], params['classes'], method='loadable')

