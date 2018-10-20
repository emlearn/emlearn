
# emtrees
Tree-based machine learning classifiers for microcontroller and embedded systems.
Train in Python, then do inference on any device with support for C.

Want Naive Bayes instead? Go to [embayes](https://github.com/jonnor/embayes)

## Key features

Embedded-friendly Inference

* Portable C99 code
* No libc required
* No dynamic allocations
* Integer/fixed-point math only
* Single header file include
* Fast, sub-millisecond classification
* Memory efficient. Can run with `<100 bytes RAM`

Convenient Training

* API-compatible with [scikit-learn](http://scikit-learn.org)
* Implemented in Python 3
* C classifier accessible in Python using pybind11

[MIT licensed](./LICENSE.md)

Can be used as an open source alternative to MATLAB Classification Trees,
Decision Trees using MATLAB Coder for C/C++ code generation.
`fitctree`, `fitcensemble`, `TreeBagger`, `ClassificationEnsemble`, `CompactTreeBagger`

## Status
**Minimally useful**

* Random Forests and ExtraTrees classifiers implemented
* Tested running on AVR Atmega, ESP8266 and Linux.
* On ESP8266, 8x8 digits classify in under 0.3ms with 95%+ accuracy
* On Linux, is approx 2x faster than sklearn

## Installing

Install from PyPI

    pip install emtrees --user

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
import emtrees
cmodel = emtrees.convert(estimator, method='inline')
code = cmodel.output_c('sonar')
with open('sonar.h', 'w') as f:
   f.write(code)
```

3. Use the C code

```c
#include "sonar.h"

const int32_t length = 60;
int32_t values[length] = { ... };
const int32_t predicted_class = sonar_predict(values, length):
```


For full example code, see [examples/digits.py](./examples/digits.py)
and [emtrees.ino](./emtrees.ino)

## TODO

0.2

* Standalone example application on microcontroller
* Include emtrees.h inline in generated code

1.0

* Support returning probabilities
* Support serializing/deserializing trees

Probably

* Support sklearn GradientBoostingClassifier
* Support regression trees
* Support weighted voting
* Implement Isolation Forests (requires path/depths)

Maybe

* Support [XGBoost](https://github.com/dmlc/xgboost) learning of trees
* Support [LightGBM](https://github.com/Microsoft/LightGBM) learning of trees
* Support [CatBoost](https://github.com/catboost/catboost) learning of trees
* Support/Implement a Very Fast Decision Tree (VFDT) learning algorithm
* Implement multithreading when used in Python bindings, using OpenMP

