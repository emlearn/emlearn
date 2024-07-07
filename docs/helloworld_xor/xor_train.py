import emlearn
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer

# Generate simple dataset
def make_xor(lower=0.0, upper=1.0, threshold=0.5, samples=100, seed=42):
    rng = numpy.random.RandomState(seed)
    X = rng.uniform(lower, upper, size=(samples, 2))
    y = numpy.logical_xor(X[:, 0] > threshold, X[:, 1] > threshold)
    # convert to int16 with a 8 bit range. [0.0-1.0] -> [0-255]
    X = (X * 255).astype(numpy.int16)
    return X, y

X, y = make_xor()

# Train a model
estimator = RandomForestClassifier(n_estimators=3, max_depth=3, max_features=2, random_state=1)
estimator.fit(X, y)
score = get_scorer('f1')(estimator, X, y)
assert score > 0.90, score # verify that we learned the function

# Convert model using emlearn
path = 'xor_model.h'
cmodel = emlearn.convert(estimator, method='inline')
cmodel.save(file=path, name='xor_model')
print('Wrote model to', path)
