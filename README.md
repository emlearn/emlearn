[![Travis CI Build Status](https://travis-ci.org/jonnor/emlearn.svg?branch=master)](https://travis-ci.org/jonnor/emlearn)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/myb325oc06w89flc?svg=true)](https://ci.appveyor.com/project/jonnor/emlearn)

# emlearn

Machine learning for microcontroller and embedded systems.
Train in Python, then do inference on any device with a C99 compiler.

## Key features

Embedded-friendly Inference

* Portable C99 code
* No libc required
* No dynamic allocations
* Support integer/fixed-point math
* Single header file include

Convenient Training

* Using Python with [scikit-learn](http://scikit-learn.org) or [Keras](https://keras.io/)
* The generated C classifier is also accessible in Python

[MIT licensed](./LICENSE.md)

Can be used as an open source alternative to MATLAB Classification Trees,
Decision Trees using MATLAB Coder for C/C++ code generation.
`fitctree`, `fitcensemble`, `TreeBagger`, `ClassificationEnsemble`, `CompactTreeBagger`

## Status
**Minimally useful**

Classifiers:

* `eml_trees`: sklearn.RandomForestClassifier, sklearn.ExtraTreesClassifier, sklearn.DecisionTreeClassifier
* `eml_net`: sklearn.MultiLayerPerceptron, Keras.Sequential with fully-connected layers
* `eml_bayes`: sklearn.GaussianNaiveBayes

Feature extraction:

* `eml_audio`: Melspectrogram

Tested running on AVR Atmega, ESP8266 and Linux.

## Installing

Install from PyPI

    pip install --user emlearn

## Usage

1. Train your model in Python

```python
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(n_estimators=10, max_depth=10)
estimator.fit(X_train, Y_train)
...
```

2. Convert it to C code
```python
import emlearn
cmodel = emlearn.convert(estimator, method='inline')
cmodel.save(file='sonar.h')
```

3. Use the C code

```c
#include "sonar.h"

const int32_t length = 60;
int32_t values[length] = { ... };
const int32_t predicted_class = sonar_predict(values, length):
```


For full example code, see [examples/digits.py](./examples/digits.py)
and [emlearn.ino](./emlearn.ino)


