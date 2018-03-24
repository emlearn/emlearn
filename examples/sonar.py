
import emtrees
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load and prepare data
filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
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

# Output C code
code = estimator.output_c('sonar')
outfile = 'sonar.h'
with open(outfile, 'w') as f:
   f.write(code)

print('Wrote C code to', outfile)
