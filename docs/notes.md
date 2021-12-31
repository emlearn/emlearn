
# emlearn examples

- Learn some generally applicable concept or technique
- Easy to reproduce
Needing no or minimal special hardware, or long setup
Easy to check that one has done it correctly. Expected results, compare
- Encourages people to try out emlearn for their problem
- Inspires people to make things with emlearn

# Learning goals

Basics

- Convert a model to C, run on computer
- Run model on microcontroller. Eg Arduino
- Optimize model. Hyperparamters, conversion parameters. Ex RandomForest
- Collect data from microcontroller. Build model on this data

Classification

- Classifier-as-detector for time-series signals
- Time-conditional modeling

Anomaly detection

- Static anomaly detection
- Anomaly detection conditional on time or context
- Combined with classifier, unknown class

TinyML benefits

- Reacting quickly. Stop system on issue.
- Save power. Only transmit data that is needed
- Standalone operation. User interface on device

ML benefits

- Handle complex phenomena.
Multi-variate. Multi-modal. Complex variable interactions. Complex time dependency.
- Data-driven development.
Structured methodology
Known cases covered. Known performance.

Misc

- Feature engineering?
Normalization
Power-transforms / box-cox

# Usage goals

- Get an initial model working
Without spending a lot of optimizing performance

- Find a model that fits within compute constraints C,
while reaching performance target T

- Find the pareto optimal model
Accuracy vs compute/energy

- Collect a dataset
Including labeling

## Feature engineering
Feature extraction/engineering critical to model performance.
Both predictive accuracy, as well as.
May need to create new features from raw data.
One approach is to generate a large set of potentially-useful candidates,
and then use Feature Selection to find a good subset.

Some auto generation approaches exist.

### catch22
https://github.com/chlubba/catch22
Defines 22 features for dynamical aspects of univariate time-series.
Has implementations in C. However using doubles and malloc.
The features are explained here:
https://feature-based-time-series-analys.gitbook.io/catch22-features/
They are relatively complex, consisting of several steps, online "estimation"
z-score, histogram, Welch method with rectangular window,
Note: no fetures for "static" aspects, such as mean,std etc

### Matrix profile
Especially for finding common motifs, or anomalies (discords)
https://matrixprofile.org/
https://stumpy.readthedocs.io/en/latest/index.html
How is it in terms of CPU and RAM requirements for typical embedded tasks?

### tsfresh
https://github.com/blue-yonder/tsfresh
Time-series feature generation, with integrated feature selection.
Extracts 100s of features automatically
Provided as scikit-learn transformers
Overview of features:
https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html

## featuretools
https://featuretools.alteryx.com/
composes a set of primitives by stacking them.
Primitives can be either transformations or aggregators
Gives new feature names that include description of calculations

### autofeat
https://github.com/cod3licious/autofeat
Linear Prediction Models with Automated Feature Engineering and Selection

## MiniRocket
MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification
August 2021
KDD '21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data MiningAugust 2021 Pages 248–257https://doi.org/10.1145/3447548.3467231
https://arxiv.org/pdf/2012.08791.pdf
https://github.com/angus924/minirocket

Transforms time-series using a set of convolution kernels.
And then classifies using a linear classifier.


MiniRocket uses a small, fixed set of kernels, and is almost entirely deterministic.
Uses dilation , sampled on an exponential scale
Uses ‘proportion of positive values’ pooling (PPV)

Weights constrained to two values
Similarities with binary and quantised convolutional neural networks

MiniRocket exploits various properties of the kernels, and of PPV,
in order to massively reduce the time required for the transform.
Precomputing the product of the kernel weights and the input,
and using those precomputed values to construct the convolution output


Trained 109 datasets from UCR in 10 minutes.
Compares against other methods
- cBOSS. dictionary methods
- Temporal Dictionary Ensemble (TDE) is a recent dictionary method
based on the frequency of occurrence of patterns in time series. TDE combines aspects of earlier dictionary methods including cBOSS
- Proximity Forest, an ensemble of decision trees using distance measures as splitting criteria
- Canonical Interval Forest (CIF) is a recent method which adapts the Time Series Forest (TSF)
to use catch22 features. CIF is significantly more accurate than either catch22 or TSF.
- TS-CHIEF builds on Proximity Forest.
In addition to distance measures, it uses interval-based and spectralbased splitting criteria [36]
- HIVE-COTE is an ensemble of other methods including BOSS and TSF.

MosquitoSound, InsectSound, FruitFlies

84 different kernels
With different dilations. 119 dialations = 10k features

linear in the number of kernels/features (𝑘),
linear the number of examples (𝑛),
and linear time series length (input),
tin total, 𝑂(𝑘 · 𝑛 · input)

2**9 = 512 possible two-valued kernels of length 9.
Uses a subset of 84 of these kernels, all combinations with 3 values of B (and 6 values of A)
Uses kernels of length 9 with 2 values, 𝛼 = −1 and 𝛽 = 2.
Choice of 𝛼 and 𝛽 is arbitrary, since the scale of these values is unimportant.
It is only important that the sum of the weights should be zero or, equivalently, that 𝛽 = −2A

PPV is bounded between 0 and 1

Take [0.25, 0.5, 0.75] quantiles from 𝑊𝑑 ∗ 𝑋 as bias values, to be used in computing PPV

Wespecify dilations in the range 𝐷 = {2**0, ..., 2**max⌋},
such that the maximum effective length of a kernel, including dilation,
is the length of the input time series.
Default limit the maximum number of dilations per kernel to 32

Padding is alternated for each kernel/dilation combination such that,
overall, half the kernel/dilation combinations use padding, and half do not.

reuse the convolution output, 𝐶, to compute multiple features, with different bias values

it is only necessary to compute 𝛼𝑋 and 𝛽𝑋 once for each input time series,
and then reuse the results to complete the convolution operation for each kernel by addition


3.2.3 Avoiding Multiplications.
Restricting the kernel weights to two values allows us to,
in effect, ‘factor out’ the multiplications from the convolution operation,
and to perform the convolution operation using only addition.


Transforms are not integrated with the classifier.
Likely overcomplete.
? how sparse are the feature weights. How many can be dropped for each problem?

? what if to use trees rather than linear methods

## Feature selection

Greedy feature selection.
Can either start from no features (forward mode),
or all features (backward).
Well supported in standard scikit-learn ecosystem:
 
- https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection
- http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

mlextend additionally supports "floating" versions of forward/backwards,
which samples a few more permutations.


## Model costs 
The model has a cost in terms of

- Execution time
- RAM memory
- FLASH program memory
- Energy

In addition to the model there is also
- Data aquisition
- Feature extraction

Measuring energy requires external tools.
TODO: make a tool that outputs

Execution time is device dependent.
And data dependent for some methods, like decision trees.
RAM,FLASH is not device dependent.

Tools for getting these values easily would be valuable
On host should be able to get RAM,FLASH very close to what will be on device.
Might have some slight compiler differences.

Execution time will be quite different.
But maybe relative speeds are quite similar?
At least within a single model type.
Differences in processor architecture might change things though.

Very practical if pretty cost metrics can be acheived without going on device.
Easier experimentation.
Can provide standard tools.
And can be integrated into automated workflows.


Could compile it for a representative device (ARM Cortex M or similar).
Can then use arm-none-eabi-size to get the FLASH and SRAM size.
Need a (device-specific) linker script to do this.

Could also use an emulator to run it, to get approximate runtimes.

Mainline Qemu only support TI Stellaris lsm3s
https://wiki.qemu.org/Documentation/Platforms/ARM
Zephyr has support for this platform
https://docs.zephyrproject.org/2.6.0/boards/arm/qemu_cortex_m3/doc/index.html

Zephyr still working on Cortex M4F Qemu, using netduinoplus2
https://github.com/zephyrproject-rtos/zephyr/issues/22870

Zephyr also use QEMU for Cortex M33 hardfp
using MPS2 AN521
https://github.com/zephyrproject-rtos/zephyr/pull/35381
https://docs.zephyrproject.org/2.6.0/boards/arm/mps2_an521/doc/index.html?highlight=an521

Since qemu 4.2.0 with the insn plugin, it is possible to count the number of instructions
https://stackoverflow.com/questions/58766571/how-to-count-the-number-of-guest-instructions-qemu-executed-from-the-beginning-t
https://qemu.readthedocs.io/en/latest/devel/tcg-icount.html

mps2-an505 	arm 	cortex-m33
https://github.com/ajblane/armv8m-hello

musca-a 	arm 	cortex-m33

Qemu support for ESP32 is in a fork by Espressif, with goal of mainlining
https://github.com/espressif/qemu/wiki
As of December 2021 the patchset is only 25 patches

Hello World tutorial
https://balau82.wordpress.com/2011/09/03/using-codesourcery-bare-metal-toolchain-for-cortex-m3/

This is however a ARM Cortex M3 chip, without floating-point support

AVR is supported in qemu, including "uno" and "mega2560" machines
https://qemu.readthedocs.io/en/latest/system/target-avr.html
Also really simple to compile, as linker scripts are included with avr-gcc
https://github.com/jarijokinen/avr-examples
https://yeah.nah.nz/embedded/qemu-avr/

Renode supports many microcontroller boards
https://renode.readthedocs.io/en/latest/introduction/supported-boards.html

# Tooling

Sphinx gallery
Generate visual outputs from runnable examples
Supports rST in Python files
https://sphinx-gallery.github.io/stable/index.html
Can generate Jupyter notebooks. And also binder links

nbsphinx is an alternative.
Requires Jupyter / ipynb format
https://nbsphinx.readthedocs.io/en/0.8.7/

Notebooks have advantage of being easy to share online. Eg with Binder
Forces use of the notebook computational and editing model
Embedded device less likely to be familiar though?

# Available materials

Microcontroller devboard.
eg Arduino Uno

Electronic components
- Resistors,capacitors,LEDs,wires
BTJ/MOSFET transistors?


# Possible examples

## XOR problem

See [XOR classifier example](https://emlearn.readthedocs.io/en/latest/auto_examples/xor.html).

Minsky, M. Papert, S. (1969). Perceptron: an introduction to computational geometry. The MIT Press, Cambridge, expanded edition, 19(88), 2.

Showed that Perceptron could not learn XOR problem because the problem is not linearly separable.

Rumelhart, D. Hinton, G. Williams, R. (1985). Learning internal representations by error propagation (No. ICS-8506). California University San Diego LA Jolla Inst. for Cognitive Science.

Showed that can be solved with backpropagation.


## Component classifier

See [emlearn-component-tester project](https://github.com/emlearn/emlearn-component-tester).

## Photovoltaic detector

Two LEDs facing eachother.
Or two LEDs. Reflective
Measure voltage across one LED
Control other LED on or with PWM.
Put LED on/off.
Use low state as reference. Compute difference with when on.
Put item in front of sensor -> should be classified as present

Can be done over time, for pass-by

Thing to try. What happens




# Ideas

## Temperature sensor

Without a specialized thermometer.

Regression task.

Diode voltage drop. Ie 1N4007. Depends almost linearly on temperature. 
Similar might also be for a Base-Emitter pair in a bipolar transistor?

Humidity. Capacitors capacitance are dependent on relative humidity.
Probably also has some temperature dependence. May need to be corrected for!

Probably has some component variation also. Would be nice to map out.

Should collect data in environmental chamber.
Using one or more standardized devices as a reference.

Minituarization. Could put on board with an attiny, act as I2C.


## Anomaly Detection

Anomaly Detection
Known relationship between two variables
Introduce an anomaly
Change capacitor. Higher/lower value
Degrate the capacitor. Eg using heat

## Cyclic behavior

Actuator that is driven. From state 1 to state 2, continious transition
Change in position should be even over time. For example a linear pattern, or linear angular
Would need to measure position... Then might also have PID.
Setpoint vs actual interesting variable to track

Could track amount of current to the motor. Should be approximately constant
Becomes more interesting with multiple motors/axes

## Activity Recognition using wrist-mounted accelerometer
Using tri-axial accelerometer.

Several standard hardware devices available.

- LilyGo TTGO T-Watch. ESP32
- Pine64 [PineTime](https://www.pine64.org/pinetime/). nRF52832, BMA425

[Energy-efficient activity recognition framework using wearable accelerometers](https://doi.org/10.1016/j.jnca.2020.102770).
Uses 50/100 hz, int8. 256 samples long windows, with 50% overlap.
Computes from this basic .
Classified with RandomForest
Gets F1 scores of 90%, 70% on participants not in training set.
Code available here, including C feature extraction, Python feature extraction and datasets.
https://github.com/atiselsts/feature-group-selection

Can maybe be extended to gesture recognition later.

Being able to store/annotate activities on the go would be great.
To build up datasets.
Chose between pre-defined classes.
Have a couple of user-definable classes.
1/2/3/4 or blue/red/green/yellow 
Pre-annotate class, before starting acitvity. 
Post-annotate after doing activity / event happened.

Should be able to store raw data from accelerometer.
Maybe use some simple compression. Like gzip/deflate
Store files to be synced as time-stamped.
Maybe one per 60 seconds or so.

On-device few-shot learning of these would also be very cool.
kNN the most simple algorithm for this.
Just need to store feature vectors somewhere. FLASH/SDCARD
And keep number managable, so not too slow things down too much.
Need to have a good feature extraction system.

DynamicTimeWarping kNN one alternative for few-shot.
https://sequentia.readthedocs.io/en/latest/sections/classifiers/knn.html
https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/
https://github.com/datashinobi/K-nearest-neighbors-with-dynamic-time-wrapping/blob/master/knndtw.py
https://github.com/MaxBenChrist/mlpy-plus-dtw/blob/master/mlpy/dtw/cdtw.c

## Condition monitoring moveable machine using accelerometer

Raise an alarm.
E.g send via WiFi to user. ESP32 etc

Accelerometer on head.

+ single sensor for motion on all 3 axis
+ standardized,commodity sensor 
+ easy to adapt to other cases of moving machinery 

Example washing machine
- Idle. No vibrations
- Running. Some vibrations
- Anomaly. Vibrating too much. One of the legs mis-adjusted

Example. 3d-printer
simple anomalies.
hitting object too hard with nozzle. eg from overextrusion



Motor current
- 3 motors, 4 wires on each.
Could one monitor all 4 wires together, or just one of each

## Object classification from cap sensors

Position 9 cap sensors in a grid
Maybe 10 x 10 cm area. 3 cm between each


## XY position from cap sensors

Place 4 sensor electrodes in a square
wires taped to a piece of paper

regression problem

maybe use cap sensing code from rebirth project


## Material identificator

light reflection
light transmission
    with IR,UV,visible
gases emitted. VOC
weight. with standardized level -> density
capacitive index?


sensors are on a U or UUU shaped PCB

put liquid into a small beaker
fill N millimeter from top
put the sensor PCB down into liquid
should have some stopper to make sure not to far down in liquid


conductivity/resistivity
capactivity
optical reflectivity
optical transmissivity
heat capacity
heat conductivity

transparent/translucent/opaque
reflective/not

juice
coca-cola
sprite
milk
oil
soap

carbonated vs non-carbonated drink
carbonated drink of different colors
alcoholic vs non-alchol drink
white vs red wine
different red wines against eachother
white rum vs brown rum

non-liquids
sand
larger vs smaller pebbles
moist sand versus dry
rice/wheat grains

