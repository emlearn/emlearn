
import emtrees

import numpy
import pandas
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

import random
random.seed(3)

# load and prepare data
filename = 'sonar.all-data.csv'
df = pandas.read_csv(filename, header=None)

target_columns = [60]
data_columns = list(set(list(df.columns)).difference(target_columns))

# convert floats to integer
df[data_columns] = (df[data_columns] * 2**16).astype(int)
df[target_columns] = df[target_columns[0]].astype('category').cat.codes

X_train, X_test, Y_train, Y_test = train_test_split(df[data_columns], df[target_columns])

# Run training
n_trees = 5
estimator = emtrees.RandomForest(n_estimators=n_trees, max_depth=10)	
estimator.fit(X_train, Y_train)
Y_pred = estimator.predict(X_test)
a = accuracy_score(Y_test, Y_pred)

print('Trees: %d' % n_trees)
print('Mean Accuracy: %.3f%%' % (a*100))

print('C classifier:\n\n', estimator.output_c('sonartrees'))

