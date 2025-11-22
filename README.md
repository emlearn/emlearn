
![Tests](https://github.com/emlearn/emlearn/actions/workflows/tests.yaml/badge.svg)
![DOI](https://zenodo.org/badge/125562741.svg)]

# emlearn

Machine learning for microcontroller and embedded systems.
Train in Python, then do inference on any device with a C99 compiler.

![emlearn logo](./brand/emlearn-logo-wordmark-wide-600px.png)

## Status
**Actively maintained since 2018**.

## License

emlearn is [MIT licensed](https://github.com/emlearn/emlearn/blob/master/LICENSE.md).

## Key features

Convert machine learning models to C

* Supports models made with [scikit-learn](http://scikit-learn.org) and [Keras](https://keras.io/)
* Supports [classification](https://emlearn.readthedocs.io/en/latest/classification.html), [regression](https://emlearn.readthedocs.io/en/latest/regression.html) and [anomaly/outlier detection](https://emlearn.readthedocs.io/en/latest/anomaly_detection.html)
* Supports [tree-based ensembles](https://emlearn.readthedocs.io/en/latest/tree_based_models.html), such as Random Forest, decision trees
* Supports simple neural networks, such as Multi-Layer Perceptron (MLP)

Embedded-friendly Inference

* Portable C99 code
* No dynamic allocations
* Small code size (from 2kB FLASH)
* Small RAM size (from 50 bytes RAM)
* Support integer/fixed-point math (some models)
* No libc required (some models)

Easy to integrate in project

* Single header file to include
* Easy to embed/integrate in other languages, via C API
* Packaged as [Arduino library](https://emlearn.readthedocs.io/en/latest/getting_started_arduino.html)
* Packaged as [Zephyr module](https://emlearn.readthedocs.io/en/latest/getting_started_zephyr.html)
* MicroPython bindings via [emlearn-micropython](https://github.com/emlearn/emlearn-micropython)

Feature extraction and data processing utilities

* Infinite Impulse Response (IIR) filters
* Fast Fourier Transform (FFT)
* Read/write CSV files, with streaming support

Model validation tools

* Access generated C classifier via Python, to verify prediction correctness
* Estimate model computational cost and size (using scikit-learn compatible metrics).
[Example](https://emlearn.readthedocs.io/en/latest/auto_examples/trees_hyperparameters.html).
* Measure tools for model program/FLASH size.
[Example](https://emlearn.readthedocs.io/en/latest/auto_examples/trees_feature_quantization.html).

## Platform support

Should work *anywhere* that has working C99 compiler.
Tested running on a large range of bare-metal, RTOS and desktop operating systems.
Such as ESP8266, ESP32, AVR Atmega (8 bit), ARM Cortex M (STM32), Linux, Mac OS and Windows.

## Projects using emlearn

emlearn has been used in many projects by many different developers,
across a range of usecases and applications.
See the [Made with emlearn](https://emlearn.readthedocs.io/en/latest/made_with.html) section in documentation.

## Model support

Classification:

* `eml_trees`: sklearn.RandomForestClassifier, sklearn.ExtraTreesClassifier, sklearn.DecisionTreeClassifier
* `eml_net`: sklearn.MultiLayerPerceptron, Keras.Sequential with fully-connected layers
* `eml_bayes`: sklearn.GaussianNaiveBayes

Regression:

* `eml_trees`: sklearn.RandomForestRegressor, sklearn.ExtraTreesRegressor, sklearn.DecisionTreeRegressor
* `eml_net`: Keras.Sequential with fully-connected layers

Unsupervised / Outlier Detection / Anomaly Detection

* `eml_distance`: sklearn.EllipticEnvelope (Mahalanobis distance)
* `eml_mixture`: sklearn.GaussianMixture, sklearn.BayesianGaussianMixture

## Documentation

For full documentation see [examples](https://emlearn.readthedocs.io/en/latest/examples.html),
the [user guide](https://emlearn.readthedocs.io/en/latest/user_guide.html).

#### Other learning resources

emlearn and emlearn-micropython has been covered in the following presentations.

- Microcontrollers + Machine Learning in 1-2-3 (PyData Global 2024).
[Slides etc](https://github.com/jonnor/embeddedml/tree/master/presentations/PyDataGlobal2024)
- Sensor data processing on microcontrollers with MicroPython and emlearn (PyConZA 2024).
[Slides etc](https://github.com/jonnor/embeddedml/tree/master/presentations/PyConZA2024)
- 6 years of open source TinyML with emlearn - a scikit-learn for microcontrollers (TinyML EMEA 2024)
[YouTube video](https://www.youtube.com/watch?v=oG7PjPMA3Is) |
[Slides etc](https://github.com/jonnor/embeddedml/tree/master/presentations/TinymlEMEA2024)
- emlearn - Machine Learning for Tiny Embedded Systems (Embedded Online Conference 2024).
[Youtube video](https://www.youtube.com/watch?v=qamVWmcBdmI) |
[Slides etc](https://github.com/jonnor/embeddedml/tree/master/presentations/EmbeddedOnlineConference2024)
- Machine Learning on microcontrollers using MicroPython and emlearn (PyCon DE & PyData Berlin 2024).
[Slides etc](https://github.com/jonnor/embeddedml/tree/master/presentations/PyDataBerlin2024) |
[YouTube video](https://www.youtube.com/watch?v=_MGm8sctqjg&t=1311s&pp=ygUSZW1sZWFybiBtaWNyb3B5dGhv).


## Installing

Install from PyPI

    pip install --user emlearn

## Usage
The basic usage consist of 3 steps:

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
cmodel.save(file='sonar.h', name='sonar')
```

3. Use the C code

### Simple classifiers
```c
#include "sonar.h"

const int32_t length = 60;
int16_t values[length] = { ... };

// using generated "inline" code for the decision forest
const int32_t predicted_class = sonar_predict(values, length):

// ALT: using the generated decision forest datastructure
const int32_t predicted_class = eml_trees_predict(&sonar, length):
```

### Neural net regressor

Copy the generated `.h` file, the `eml_net.h` and `eml_common.h` into your project, then

```c
#include "nnmodel.h" // the generated code basedon on keras.Sequential

float values[6] = { ... };

const float_t predicted_value = nnmodel_regress1(values, 6);
if (predicted_value == NAN) {
    exit(-1);
}
// Process the value as needed

// Or, passing in a result array directly if more than 1 output is generated
float out[2];
EmlError err = nnmodel_regress(values, 6, out, 2);
if (err != EmlOk)
{
    // something went wrong
}
else {
    // predictions are in the out array
}
```

For a complete runnable code see [Getting Started](https://emlearn.readthedocs.io/en/latest/getting_started_host.html).


## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines,
and [docs/developing.md](docs/developing.md) for how to develop.


## Contributors

```
Jon Nordby
Mark Cooke
```

## Citations

If you use `emlearn` in an academic work, please reference it using:

```tex
@misc{emlearn,
  author       = {Nordby, Jon AND Cooke, Mark AND Horvath, Adam},
  title        = {{emlearn: Machine Learning inference engine for 
                   Microcontrollers and Embedded Devices}},
  month        = mar,
  year         = 2019,
  doi          = {10.5281/zenodo.2589394},
  url          = {https://doi.org/10.5281/zenodo.2589394}
}
```


