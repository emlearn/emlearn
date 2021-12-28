
"""
Classifier comparison
===========================

Simple demonstration of the different implemented classifiers in emlearn
"""

import os.path

import emlearn
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

try:
    # When executed as regular .py script
    here = os.path.dirname(__file__)
except NameError:
    # When executed as Jupyter notebook / Sphinx Gallery
    here = os.getcwd()

# %%
# Create dataset
# ------------------------
#
# Using a simple multi-class dataset included with scikit-learn
def load_dataset():
    from sklearn import datasets
    data = datasets.load_wine(as_frame=True)

    df = data.data.copy()
    df.columns = data.feature_names
    df['target'] = data.target

    return df

dataset = load_dataset()

# %%
# Train, convert and run model
# ------------------------------
#
# Using a simple dataset included with scikit-learn
def build_run_classifier(model, name):
    from sklearn.model_selection import train_test_split

    target_column = 'target'

    # Train model
    test, train = train_test_split(dataset, test_size=0.3, random_state=3)
    feature_columns = list(set(dataset.columns) - set([target_column]))

    model.fit(train[feature_columns], train[target_column])

    temp_dir = os.path.join(here, 'classifiers')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    model_filename = os.path.join(temp_dir, f'{name}_model.h')
    cmodel = emlearn.convert(model)
    code = cmodel.save(file=model_filename, name='model')
    
    print('Generated', model_filename)
    # TODO: generate a test dataset
    # TODO: generate a test function, verifying perf on test dataset


# %%
# Run all classifiers
# --------------------------------
#
# Using a simple dataset included with scikit-learn
import sklearn.ensemble
import sklearn.tree
import sklearn.neural_network
import sklearn.naive_bayes

classifiers = {
    'random_forest': sklearn.ensemble.RandomForestClassifier(n_estimators=10),
    'extra_trees': sklearn.ensemble.ExtraTreesClassifier(n_estimators=10), 
    'decision_tree': sklearn.tree.DecisionTreeClassifier(),
    'sklearn_mlp': sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=30),
    'gaussian_naive_bayes': sklearn.naive_bayes.GaussianNB(),
}

for name, cls in classifiers.items():
    build_run_classifier(cls, name)


