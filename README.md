
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
* C classifier accessed via pybind11

[MIT licensed](./LICENSE.md)

## Status
**Proof-of-concept**

Binary classification using Random Forests is implemented.
Tested running on AVR, ESP8266 and Linux.

## Installing

Install from git

    git clone https://github.com/jonnor/emtrees
    cd emtrees
    pip install ./


## Usage
For now, see the [tests](./tests)


## TODO

0.1

* Investigate/fix both yes+no leading to same class
* Release as Python library on PyPI

0.2

* Support multi-target classification
* Add validation to performance benchmarks
* Run tests on/against microcontroller
* Optimize training time

1.0

* Support generating specialized C/machine code for trees
* Support serializing/deserializing trees
* Support storing trees in PROGMEM
* Implement Extratrees learning
