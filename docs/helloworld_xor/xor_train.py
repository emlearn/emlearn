import emlearn
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer

# Generate simple dataset
def make_xor(lower=0.0, upper=1.0, threshold=0.5, samples=100, seed=42):
    rng = numpy.random.RandomState(seed)
    X = rng.uniform(lower, upper, size=(samples, 2))
    y = numpy.logical_xor(X[:, 0] > threshold, X[:, 1] > threshold)
    return X, y

X, y = make_xor(samples=10000, upper=0.9)

# Train a model
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(max_features=None, max_depth=3, random_state=6, criterion='gini', splitter='best', ccp_alpha=0.01)
#estimator = RandomForestClassifier(n_estimators=2, max_depth=3, max_features=2, random_state=1)
estimator.fit(X, y)
score = get_scorer('f1')(estimator, X, y)

# Convert model using emlearn
path = 'xor_model.h'
#cmodel = emlearn.convert(estimator, method='inline')
#cmodel.save(file=path, name='xor_model')
print('Wrote model to', path)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig = plt.figure(figsize=(15, 10))
plot_tree(estimator, 
          feature_names=['A', 'B'],
          class_names=['0', '1'], 
          filled=True, impurity=True, 
          rounded=True)
fig.tight_layout()
fig.savefig('xor_model.png')
estimator

assert score > 0.90, score # verify that we learned the function
