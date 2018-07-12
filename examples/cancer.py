

import emtrees
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

# load and prepare data
data = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

# Run training
n_trees = 5
estimator = emtrees.GradientBoostingClassifier(n_estimators=n_trees, max_depth=10)	
estimator.fit(X_train, Y_train)
Y_pred = estimator.predict(X_test)
a = accuracy_score(Y_test, Y_pred)

print('Trees: %d' % n_trees)
print('Mean Accuracy: %.3f%%' % (a*100))
