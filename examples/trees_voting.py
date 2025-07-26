#!/usr/bin/env python
# coding: utf-8

"""
Voting options for tree-based models
===========================

This example illustrates hard majority voting
vs soft voting options for trees.
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
# Train a RandomForest model
# ------------------------
#
# Key thing is to transform the data into integers that fit the
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, get_scorer

def prepare_data(data, label_column = 'label'):
    feature_columns = list(set(data.columns) - set([label_column]))
    X = data[feature_columns]
    Y = data[label_column]

    # Rescale and convert to integers (quantize)
    # Here everything is made to fit in int16
    X = (MinMaxScaler().fit_transform(X) * 2**15-1).astype(int)

    return X, Y

def train_model(data):

    X, Y = prepare_data(data)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)

    # sanity check performance
    cv = StratifiedKFold(5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='roc_auc')
    assert numpy.mean(scores) >= 0.75, (numpy.mean(scores), scores)

    model.fit(X_train, Y_train)

    Y_pred = model.predict_proba(X_test)[:, 1]
    test_score =  average_precision_score(Y_test, Y_pred, pos_label=pos_label)
    assert test_score >= 0.75, test_score

    return model, X_test, Y_test

from emlearn.examples.datasets.sonar import load_sonar_dataset
data = load_sonar_dataset()
pos_label = 'rock'

#from sklearn.datasets import fetch_openml
#data, y = fetch_openml('heart-statlog', version=1, as_frame=True, return_X_y=True)
#data['label'] = y
#pos_label = 'present'

print(data.head())
model, X_test, Y_test = train_model(data)

# %%
# Run experiments with different leaf_bits settings 
# -------------------------------
#
# 
from emlearn.evaluate.trees import model_size_bytes, model_size_leaves


def run_experiment(leaf_bits):

    # Do conversion with specified leaf_bits
    c_model = emlearn.convert(model, method='loadable', leaf_bits=leaf_bits)
    #model_code = c_model.save(name=model_name, include_proba=True)

    # As a reference, compute the score before conversion
    Y_pred_ref = model.predict_proba(X_test)[:, 1]
    ref_score = average_precision_score(Y_test, Y_pred_ref, pos_label=pos_label)

    # Estimate predictive performance after conversion
    Y_pred = c_model.predict_proba(X_test)[:, 1]
    score = average_precision_score(Y_test, Y_pred, pos_label=pos_label)

    # Estimate model size
    model_size = model_size_bytes(c_model)
    model_leaves = model_size_leaves(c_model)

    out = pandas.Series({
        'model_size_bytes': model_size,
        'model_leaves': model_leaves,
        'leaf_bits': leaf_bits,
        'score': round(100*score, 2),
        'ref_score': round(100*ref_score, 2),
    })

    return out


experiments = pandas.DataFrame({
    'leaf_bits': [0,2,3,4,5,6,7,8],
})
results = experiments.leaf_bits.apply(run_experiment)
print(results)



# %%
# Plot results
# -------------------------------
#
# There can be considerable reductions in program memory consumption
# by picking a suitable datatype for the platform.

def plot_results(results):
    results = results.reset_index()
    results['name'] = results.platform + '/' + results.cpu

    g = seaborn.catplot(data=results,
        kind='bar',
        y='flash',
        x='dtype',
        row='name',
        height=4,
        aspect=2,
    )
    fig = g.figure
    fig.suptitle("Model size vs feature datatype")

    return fig

# TODO: implement
#fig = plot_results(results)
#fig.savefig('example-trees-voting-leafbits.png')


