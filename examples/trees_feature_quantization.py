#!/usr/bin/env python
# coding: utf-8

"""
Feature data-type in tree-based models
===========================

Tree-based models in emlearn supports both float and integer datatypes for the feature datatype.
This example illustrates how this can impact model size.
"""

import os.path
import shutil

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
from sklearn.model_selection import cross_val_score

def train_model(data):

    label_column = 'label'
    feature_columns = list(set(data.columns) - set([label_column]))
    X = data[feature_columns]
    Y = data[label_column]

    # Rescale and convert to integers (quantize)
    # Here everything is made to fit in int8, the smallest representation
    # it may be needed to adapt to larger ones, such as uint16
    X = (MinMaxScaler().fit_transform(X) * 127).astype(int)

    model = RandomForestClassifier(n_estimators=10)

    # sanity check performance
    scores = cross_val_score(model, X, Y)
    assert numpy.mean(scores) >= 0.60, numpy.mean(scores)

    model.fit(X, Y)

    return model

from emlearn.examples.datasets.sonar import load_sonar_dataset

data = load_sonar_dataset()
model = train_model(data)

# %%
# Measure how feature datatype impacts program size
# -------------------------------
#
# We are testing here on the AVR8 platform, which has no floating point unit (FPU)
# Other platforms may show different results.

from emlearn.evaluate.size import get_program_size

def check_program_size(dtype, model):

    model_name = 'sizecheck'
    features_length = model.estimators_[0].n_features_in_
    model_enabled = 0 if dtype == 'no-model' else 1

    if model_enabled:
        # Quantize with the specified dtype
        c_model = emlearn.convert(model, dtype=dtype)
        model_code = c_model.save(name=model_name, inference=['inline'])
    else:
        model_code = ""

    test_program = \
    f"""
    #include <stdbool.h>
    #include <avr/io.h>
    #include <util/delay.h>

    #if {model_enabled}
    {model_code}
    const {dtype} features[{features_length}] = {{0, }};
    #endif

    int main()
    {{
        // set PINB0 to output in DDRB
        DDRB |= 0b00000001;

        #if {model_enabled}
        const uint8_t out = {model_name}_predict(features, {features_length});
        #else
        const uint8_t out = 0;
        #endif

        // set output
        PORTB = out;
        _delay_ms(50);
    }}
    """
    data = get_program_size(test_program, platform='avr')

    return pandas.Series(data)
    

results_file = os.path.join(here, 'trees-feature-quantization-avr8.csv')
# check if AVR build tools are present. If not, just load results from a file
have_avr_buildtools = shutil.which('avr-size')
if have_avr_buildtools:
    experiments = pandas.DataFrame({
        'dtype': ('no-model', 'float', 'int32_t', 'int16_t', 'int8_t', 'uint8_t'),
    })
    results = experiments['dtype'].apply(check_program_size, model=model)
    results = pandas.merge(experiments, results, left_index=True, right_index=True)
    results = results.set_index('dtype')
    # subtract overall program size to get only model size
    results = (results - results.loc['no-model'])
    results = results.drop(index='no-model')
    results.to_csv(results_file)
    print("Ran experiments. Results written to", results_file)
else:
    print("WARNING: AVR GCC toolchain not found. Loading cached results")
    results = pandas.read_csv(results_file)

print(results)


# %%
# Plot results
# -------------------------------
#
# There can be considerable reductions in program memory consumption
# by picking a suitable datatype for the platform.

def plot_results(results):

    fig, ax1 = plt.subplots(1, figsize=(6, 6))

    seaborn.barplot(ax=ax1,
        data=results.reset_index(),
        y='program',
        x='dtype',
    )
    fig.suptitle("Model size vs feature datatype (platform=AVR8)")

    return fig

fig = plot_results(results)
fig.savefig('example-trees-feature-quantization.png')


