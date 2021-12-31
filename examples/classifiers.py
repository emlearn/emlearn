
"""
Classifier comparison
===========================

Simple demonstration of the different implemented classifiers in emlearn
"""

import os.path

import emlearn
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn

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
# Using the standard scikit-learn process,
# and then using emlearn to convert the model to C
def build_run_classifier(model, name):
    from sklearn.model_selection import train_test_split

    target_column = 'target'

    # Train model
    test, train = train_test_split(dataset, test_size=0.3, random_state=3)
    feature_columns = list(set(dataset.columns) - set([target_column]))

    model.fit(train[feature_columns], train[target_column])

    out_dir = os.path.join(here, 'classifiers')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_filename = os.path.join(out_dir, f'{name}_model.h')
    cmodel = emlearn.convert(model)
    code = cmodel.save(file=model_filename, name='model')
    
    test_pred = cmodel.predict(test[feature_columns])

    # Generate a test dataet
    test_data = numpy.array(test[feature_columns]).flatten()
    test_res = numpy.array(test_pred).flatten()

    test_dataset = "\n".join([
        emlearn.cgen.array_declare(f"{name}_testset_data", dtype='float', values=test_data),
        emlearn.cgen.array_declare(f"{name}_testset_results", dtype='int', values=test_res),
        emlearn.cgen.constant_declare(f'{name}_testset_features', val=len(feature_columns)),
        emlearn.cgen.constant_declare(f'{name}_testset_samples', val=len(test)),
    ])

    test_code = test_dataset + \
    f'''
    #include "{model_filename}" // emlearn generated model

    #include <stdio.h> // printf

    int
    {name}_test() {{
        const int n_features = {name}_testset_features;
        const int n_testcases = {name}_testset_samples;

        int errors = 0;

        for (int i=0; i<n_testcases; i++) {{
            const float *features = {name}_testset_data + (i*n_features);
            const int expect_result = {name}_testset_results[i*1];

            const int32_t out = model_predict(features, n_features);

            if (out != expect_result) {{
                printf(\"test-fail sample=%d expect=%d got=%d \\n\", i, expect_result, out);
                errors += 1;
            }}

        }}
        return errors;
    }}

    int
    main(int argc, const char *argv[])
    {{
        const int errors = {name}_test();
        return errors;
    }}'''

    test_source_file = os.path.join(out_dir, f'test_{name}.c')
    with open(test_source_file, 'w') as f:
        f.write(test_code)

    print('Generated', test_source_file)

    include_dirs = [ emlearn.includedir ]
    test_executable = emlearn.common.compile_executable(
            test_source_file,
            out_dir,
            name='test_{name}',
            include_dirs=include_dirs
    )

    import subprocess
    errors = None
    try:
        subprocess.check_output([test_executable])
        errors = 0
    except subprocess.CalledProcessError as e:
        errors = e.returncode

    print(f"Tested {name}: {errors} errors")

# %%
# Run all classifiers
# --------------------------------
#
# Some of the supported modela and configurations
import sklearn.ensemble
import sklearn.tree
import sklearn.neural_network
import sklearn.naive_bayes

classifiers = {
    'random_forest': sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=1),
    'extra_trees': sklearn.ensemble.ExtraTreesClassifier(n_estimators=10, random_state=1), 
    'decision_tree': sklearn.tree.DecisionTreeClassifier(),
    'sklearn_mlp': sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=30, random_state=1),
    'gaussian_naive_bayes': sklearn.naive_bayes.GaussianNB(),
}

for name, cls in classifiers.items():
    build_run_classifier(cls, name)


