
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
DynamicTimeWarping one alternative.

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

