[![Travis CI Build Status](https://travis-ci.org/emlearn/emlearn.svg?branch=master)](https://travis-ci.org/emlearn/emlearn)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/myb325oc06w89flc?svg=true)](https://ci.appveyor.com/project/emlearn/emlearn)
[![DOI](https://zenodo.org/badge/125562741.svg)](https://zenodo.org/badge/latestdoi/125562741)

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

[MIT licensed](https://github.com/emlearn/emlearn/blob/master/LICENSE.md)

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

Tested running on AVR Atmega, ESP8266, Linux and Windows.
Mac OS should also work fine, please file a bug report if it does not.

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


For full example code, see [examples/digits.py](https://github.com/emlearn/emlearn/blob/master/examples/digits.py)

## Contributors

```
Jon Nordby
Mark Cooke
```

## Citations

If you use `emlearn` in an academic work, please reference it using:

```tex
@misc{emlearn,
  author       = {Jon Nordby},
  title        = {{emlearn: Machine Learning inference engine for 
                   Microcontrollers and Embedded Devices}},
  month        = mar,
  year         = 2019,
  doi          = {10.5281/zenodo.2589394},
  url          = {https://doi.org/10.5281/zenodo.2589394}
}
```

## Made with emlearn

`emlearn` has been used in the following works.

- [An End-to-End Framework for Machine Learning-Based Network Intrusion Detection System](https://www.researchgate.net/publication/353590312_An_End-to-End_Framework_for_Machine_Learning-Based_Network_Intrusion_Detection_System) by Gustavo de Carvalho Bertoli et.al.
Implemented a TCP Scan detection system.
It used a Decision Tree and used emlearn to generate code for a Linux Kernel Module / Netfilter to do the detection.
It was tested on a Rasperry PI 4 single-board-computer, and the performance overhead was found to be negligble.
- [Towards an Electromyographic Armband: an Embedded Machine Learning Algorithms Comparison](https://webthesis.biblio.polito.it/17000/)
by Danilo Demarchi, Paolo Motto Ros, Fabio Rossi and Andrea Mongardi.
Detected different hand gestures based on ElectroMyoGraphic (sEMG) data.
Compared the performance of different machine learning algorithms, from emlearn and Tensorflow Lite.
Found emlearn RandomForest and Naive Bayes to give good accuracy with very good power consumption.
- [TinyML-Enabled Frugal Smart Objects: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9166461) by Ramon Sanchez-Iborra and Antonio F. Skarmeta.
Created a model for automatically selecting which radio transmission method to use in an IoT device.
Running on Arduino Uno (AVR8) device.
Tested Multi-layer Perceptron, Decision Tree and Random Forest from emlearn.
Compared performance with sklearn-porter, and found that Random Forest to be faster in emlearn,
while Decision Tree faster in sklearn-porter.
Compared emlearn MLP to MicroMLGen’s SVM, and found the emlearn MLP to be more accurate and lower inference time.
- [A Machine Learning Approach for Real Time Android Malware Detection](https://ieeexplore.ieee.org/abstract/document/9140771) by Ngoc C. Lê et al.
Created a C++ model for detecting malware.
Used a set of hand-engineered features and a Random Forest from emlearn as the classifier.
Running on Android devices.
- [RIOT OS](https://www.riot-os.org/) has a package for emlearn.
[RIOT OS emlearn package example](https://github.com/RIOT-OS/RIOT/tree/master/tests/pkg_emlearn).
Their build system automatically runs this test on tens of different hardware boards.

If you are using emlearn, let us know!
You can for example submit a pull request for inclusion in this README,
or create an issue on Github.


