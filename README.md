
![Tests](https://github.com/emlearn/emlearn/actions/workflows/tests/badge.svg)
[![DOI](https://zenodo.org/badge/125562741.svg)](https://zenodo.org/badge/latestdoi/125562741)

# emlearn

Machine learning for microcontroller and embedded systems.
Train in Python, then do inference on any device with a C99 compiler.

![emlearn logo](./brand/emlearn-logo-wordmark-wide-600px.png)

## Status
**Minimally useful**.
Used in dozens of projects by dozens of developers.

## Key features

Embedded-friendly Inference

* Portable C99 code
* No libc required
* No dynamic allocations
* Single header file include
* Support integer/fixed-point math (some methods)
* Can be embedded/integrated with other languages via C API

Convenient Training

* Using Python with [scikit-learn](http://scikit-learn.org) or [Keras](https://keras.io/)
* The generated C classifier is also accessible in Python

[MIT licensed](https://github.com/emlearn/emlearn/blob/master/LICENSE.md)

Supporting libraries

- [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).
Efficient Machine Learning engine for [MicroPython](https://micropython.org), using emlearn.

Can be used as an open source alternative to MATLAB Classification Trees,
Decision Trees using MATLAB Coder for C/C++ code generation.
`fitctree`, `fitcensemble`, `TreeBagger`, `ClassificationEnsemble`, `CompactTreeBagger`


## Model support

Classification:

* `eml_trees`: sklearn.RandomForestClassifier, sklearn.ExtraTreesClassifier, sklearn.DecisionTreeClassifier
* `eml_net`: sklearn.MultiLayerPerceptron, Keras.Sequential with fully-connected layers
* `eml_bayes`: sklearn.GaussianNaiveBayes

Regression:

* `eml_trees`: sklearn.RandomForestRegressor, sklearn.ExtraTreesRegressor, sklearn.DecisionTreeRegressor
* `eml_net`: Keras.Sequential with fully-connected layers (```emlearn.convert(model, method='loadable', return_type='regressor')```)

Unsupervised / Outlier Detection / Anomaly Detection

* `eml_distance`: sklearn.EllipticEnvelope (Mahalanobis distance)
* `eml_mixture`: sklearn.GaussianMixture, sklearn.BayesianGaussianMixture

Feature extraction:

* `eml_audio`: Melspectrogram

## Platform support

Tested running on AVR Atmega, ESP8266, ESP32, ARM Cortex M (STM32), Linux, Mac OS and Windows.

Should work *anywhere* that has working C99 compiler. 

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
float values[length] = { ... };

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

For full documentation see [examples](https://emlearn.readthedocs.io/en/latest/examples.html),
the [user guide](https://emlearn.readthedocs.io/en/latest/user_guide.html).

## Contributing

Check out the source code, make sure you install the `Unity` submodule as well with `git submodule update --init`

Before committing any code, run the tests by `./test.sh` and install the module locally with `pip install ./ -v`

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

## Made with emlearn

`emlearn` has been used in the following works (among others).

If you are using emlearn, let us know!
You can for example submit a pull request for inclusion in this README,
or create an issue on Github.

- [IoT Next Generation Smart Grid Meter (NGSM) for On-Edge Household Appliances Detection Based on Deep Learning and Embedded Linux](https://www.researchgate.net/publication/375226746_IoT_Next_Generation_Smart_Grid_Meter_NGSM_for_On-Edge_Household_Appliances_Detection_Based_on_Deep_Learning_and_Embedded_Linux) by Noor El-Deen M. Mohamed et. al at Helwan University in Cairo, Egypt.
Developed a smart grid meter for households that can detect when different appliances are running.
This is done using a Energy Disaggregation / Non-Intrusive Load Monitoring (NILM) model,
which implemented using a neural network.
The system runs on Embedded Linux using a Allwinner F1C200s system-on-chip.
Used emlearn instead of Tensorflow Lite to have a more light-weight approach.
- [C-HAR: Compressive Measurement-Based Human Activity Recognition](https://www.researchgate.net/publication/371821553_C-HAR_Compressive_Measurement-Based_Human_Activity_Recognition) by Billy Dawton et. al.
Tested using compressive sensing with only 5 Hz samplerate do recognize actions such as "Walking", "Typing", and "Eating".
Used emlearn to deploy the RandomForest based models to a Teensy 4.1 board.
Found that they could reach around 90% accuracy, but with 4 times lower sampling rate,
and 2 times lower execution time compared to existing compressed sensing approaches.
- [Tiny Machine Learning for Real-time Postural Stability Analysis](https://raw.githubusercontent.com/veysiadn/veysiadn.github.io/dcdcd8be5b4b3e1509c5e264501f02259b59da78/assets/pdf/1570901064%20final.pdf) by Veysi Adın et.al.
Tested an sway analysis algorithm for deploying to on a Nordic NRF52840 (ARM Cortex M4).
Compared artificial neural network (ANN) model with Random Forests and Gaussian Naive Bayes.
Found that ANN had the best performance under lower signal-to-noise ratios,
but that Random Forest had the lowest inference time and similar performance in high SNR cases.
- [Micro Random Forest: A Local, High-Speed Implementation of a Machine-Learning Fault Location Method for Distribution Power Systems](https://www.researchgate.net/publication/372482984_Micro_Random_Forest_A_Local_High-Speed_Implementation_of_a_Machine-Learning_Fault_Location_Method_for_Distribution_Power_Systems) by Miguel Jimenez Aparicio et.al at Sandia National Laboratories.
Developed a fault localization method that uses the signature of the travelling wave.
Tested 4 different sized RandomForest models, evaluted performance on a simulated power network.
Used emlearn to port the models to the TMS320F28379D chip, a C2000-series DSP from Texas Instruments.
Found that the total execution time was 1.2 ms, of which only 10 us was used by the classifier.
- [Remote Breathing Rate Tracking in Stationary Position Using the Motion and Acoustic Sensors of Earables](https://www.researchgate.net/publication/369439475_Remote_Breathing_Rate_Tracking_in_Stationary_Position_Using_the_Motion_and_Acoustic_Sensors_of_Earables) by Tousif Ahmed et.al at Samsung Research.
Developed a system using microphone and accelerometer on earbuds to estimate breathing rate of wearer.
Tested various models such as Logistic Regression, Multi-layer Perceptron and Random Forest.
Used emlearn to convert the model to C and run on Samsung Galaxy Buds 2 earbuds.
Found that the battery consumption was low enough that it could run continiously.
- [Smart Attack Detection for IoT Networks](https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=21&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=sv&pid=diva2%3A1726212&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-3227) by Yang Yang.
Implemented a Intrusion Detection System for IoT networks.
Used Random Forest classifier running on Nordic nRF52840 using Contiki-NG RTOS.
In addition to the on-device inference, they also ran the classifiers in the Cooja IoT device network simulator.
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
- [LPWAN and Embedded Machine Learning as Enablers for the Next Generation of Wearable Devices](https://www.mdpi.com/1424-8220/21/15/5218) by Ramon Sanchez-Iborra.
Evaluated feasibility of running TinyML models on a LoraWAN sensor node.
Used an ATmega 328p, with MPU6050 IMU, GY-NEO6MV2 GPS and RN2483 LoRaWAN tranceiver.
Found that code for communicating with the pheripherals took considerably more SRAM/FLASH than ML model.
Was able to fit a Random Forest with 50 trees (FLASH bound), or a multi-layer perceptron with 5 layers and 10 neurons-per-layer (SRAM bound).
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
[RIOT OS emlearn package example](https://github.com/RIOT-OS/RIOT/tree/master/tests/pkg/emlearn).
Their build system automatically runs this test on tens of different hardware boards.




