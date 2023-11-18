#!/usr/bin/env python
# coding: utf-8

"""
Compare data-types for features in tree-based models
===========================

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


from emlearn.examples.datasets.sonar import load_sonar_dataset

# %%
# Load dataset
# ------------------------
#

data = load_sonar_dataset()


# %%
# Train a RandomForest model
# ------------------------
#
from sklearn.ensemble import RandomForestClassifier

def train_model(data):

    label_column = 'label'
    feature_columns = list(set(data.columns) - set([label_column]))
    # FIXME: convert data to integers
    X = data[feature_columns]
    Y = data[label_column]

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, Y)

    return model

model = train_model(data)

# %%
# Check effect of changing dtype on program size
# -------------------------------
#

# custom metrics for model costs
from emlearn.evaluate.size import get_program_size

# FIXME: check if AVR build tools are present. If not, just load results from a file

def check_program_size(dtype, model):

    model_name = 'sizecheck'
    features_length = model.estimators_[0].n_features_in_
    model_enabled = 0 if dtype is None else 1

    if model_enabled:
        # Quantize with the specified dtype
        c_model = emlearn.convert(model, dtype=dtype)
        model_code = c_model.save(name=model_name, inference=['inline'])
    else:
        model_code = ""

    #print(model_code)

    # FIXME: check performance

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
        PORTB &= 0b11111110 + out;
        _delay_ms(50);
    }}
    """
    data = get_program_size(test_program, platform='avr')

    return pandas.Series(data)
    

experiments = pandas.DataFrame({
    'dtype': (None, 'float', 'int32_t', 'int16_t', 'int8_t', 'uint8_t'),
})
results = experiments['dtype'].apply(check_program_size, model=model)
results = pandas.merge(experiments, results, left_index=True, right_index=True)

print(results)

# FIXME: plot the results. Program size versus dtype. Mark clearly with AVR8 platform

