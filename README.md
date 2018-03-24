
# emtrees
Tree-based machine learning classifiers for embedded systems.
Train in Python, deploy on microcontroller.

## Key features

Embedded-friendly Classifier

* Portable C99 code
* No stdlib required
* No dynamic allocations
* Integer/fixed-point math only
* Single header file, less than 100 lines

Convenient Training

* API-compatible with [scikit-learn](http://scikit-learn.org)
* Implemented in Python 3
* C classifier accessible in Python using pybind11

[MIT licensed](./LICENSE.md)

## Status
**Proof-of-concept**

* Random Forests and ExtraTrees classifiers implemented
* Tested running on AVR, ESP8266 and Linux.

## Installing

Install from PyPI

    pip install emtrees --user

## Usage

1. Train your model in Python

```python
import emtrees
estimator = emtrees.RandomForest(n_estimators=10, max_depth=10)
estimator.fit(X_train, Y_train)
...
```

2. Generate C code
```python
code = estimator.output_c('sonar')
with open('sonar.h', 'w') as f:
   f.write(code)
```

3. Use the C code

```c
#include "sonar.h"

const int32_t length = 60;
EmtreesValue values[length] = { ... };
const int32_t predicted_class = sonar_predict(values, length):
```

For full example code, see [examples/sonar.py](./examples/sonar.py)

## TODO

0.2

* Add validation to performance benchmarks
* Run tests on/against microcontroller

1.0

* Support serializing/deserializing trees
* Support multi-target classification
