
![Tests](https://github.com/emlearn/emlearn/actions/workflows/tests/badge.svg)
[![DOI](https://zenodo.org/badge/125562741.svg)](https://zenodo.org/badge/latestdoi/125562741)

# emlearn

Machine learning for microcontroller and embedded systems.
Train in Python, then do inference on any device with a C99 compiler.

## Key features

Embedded-friendly Inference

* Portable C99 code
* No libc required
* No dynamic allocations
* Single header file include
* Support integer/fixed-point math (some methods)

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

Unsupervised / Outlier Detection / Anomaly Detection

* `eml_distance`: sklearn.EllipticEnvelope (Mahalanobis distance)

Feature extraction:

* `eml_audio`: Melspectrogram

Tested running on AVR Atmega, ESP8266, ESP32, ARM Cortex M (STM32), Linux, Mac OS and Windows.

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
cmodel.save(file='sonar.h')
```

3. Use the C code

```c
#include "sonar.h"

const int32_t length = 60;
int32_t values[length] = { ... };
const int32_t predicted_class = sonar_predict(values, length):
```


For full code see [the examples](https://emlearn.readthedocs.io/en/latest/examples.html).

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

- [](


We developed an initial decision tree using Gini impurity to measure the quality
of decisions, and a set of ten standardized features stored as 16-bit fixed point values. We
used cost complexity pruning [27] to limit the complexity of our decision trees and avoid
overfitting training data. Through tenfold cross-validation, we determined pruning trees
using a pruning parameter of 1.062 x 10-4, resulted in 353 nodes and the best
classification performance. We trained the decision tree model using the scikit-learn
library [28] in python and ported it to C using the emlearn library [29].

classify cattle behaviors using raw accelerometer data as the input
standing, lying, grazing, walking and ruminating

Used emlearn to
feature extraction and standardization to .

- [Power Efficient Wireless Sensor Node through Edge Intelligence](https://ieeexplore.ieee.org/document/9937324)
by Abhishek P. Damle et al.
Used accelerometer data on a wirelesess sensor node to classify the behaviour of grazing cattle,
into Standing, Grazing, Walking, Lying and Ruminating.
Used emlearn to compile a decision tree for deploying to the Microchip WLR089U0 module
(ATSAMR34x microcontroller with integrated LoRa transceiver).
The best features were selected using recursive feature elimination (RFE),
cost complexity pruning was used to tune the complexity of the decision trees.
They show that the energy required to transmit goes went down by 50 times
by doing feature extraction and classification on-edge compared to sending the raw sensor data.
- [A Comparison between Conventional and User-Intention-Based Adaptive Pushrim-Activated Power-Assisted Wheelchairs](https://www.researchgate.net/publication/356363774_A_Comparison_Between_Conventional_and_User-Intention-Based_Adaptive_Pushrim-Activated_Power-Assisted_Wheelchairs)
by M. Khalili, G. Kryt, H.F.M. Van der Loos, and J.F. Borisoff.
Implemented a user intention estimation for wheelchairs,
in order to give the user a personalized power-assist controlled.
Used emlearn to run the RandomForest classifier on a Teensy microcontroller.
Found that the real-time microcontroller model performed similar to the offline models.
- [C-AVDI: Compressive Measurement-Based Acoustic Vehicle Detection and Identification](https://www.researchgate.net/publication/356707239_C-AVDI_Compressive_Measurement-Based_Acoustic_Vehicle_Detection_and_Identification)
by Billy Dawton et.al. Implemented detection and classification of passing motorcycles and cars from sound.
Used compressed sensing system using an analog frontend and ADC running at a low samplerate.
Used a emlearn RandomForest on a Teensy microcontroller to perform the classification.
- [An End-to-End Framework for Machine Learning-Based Network Intrusion Detection System](https://www.researchgate.net/publication/353590312_An_End-to-End_Framework_for_Machine_Learning-Based_Network_Intrusion_Detection_System) by Gustavo de Carvalho Bertoli et.al.
Implemented a TCP Scan detection system.
It used a Decision Tree and used emlearn to generate code for a Linux Kernel Module / Netfilter to do the detection.
It was tested on a Rasperry PI 4 single-board-computer, and the performance overhead was found to be negligble.
- [Towards an Electromyographic Armband: an Embedded Machine Learning Algorithms Comparison](https://webthesis.biblio.polito.it/17000/)
by Danilo Demarchi, Paolo Motto Ros, Fabio Rossi and Andrea Mongardi.
Detected different hand gestures based on ElectroMyoGraphic (sEMG) data.
Compared the performance of different machine learning algorithms, from emlearn and Tensorflow Lite.
Found emlearn RandomForest and Naive Bayes to give good accuracy with very good power consumption.
- [Who is wearing me? TinyDL‐based user recognition in constrained personal devices](https://doi.org/10.1049/cdt2.12035) by Ramon Sanchez-Iborra and Antonio F. Skarmeta.
Used emlearn to implement a model for detecting who is wearing a particular wearable device, by analyzing accelerometer data.
A multi-layer perceptron was used, running on AVR ATmega328P.
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


