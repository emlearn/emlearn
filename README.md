
# emtrees
Tree-based machine learning classifiers for embedded systems.
Train in Python, deploy on microcontroller.


## Status
Proof-of-concept

Binary classification using Random Forests is implemented.

## Design

Classifier

* Portable C99 code
* No stdlib required
* No dynamic allocations
* Integer/fixed-point math only
* Single header file, less than 100 lines

Training

* Implemented in Python
* API-compatible with [scikit-learn](http://scikit-learn.org)
* C classifier accessed via pybind11

## TODO

0.1

* Release as Python library on PyPI

0.2

* De-duplicate leaf nodes
* Support multi-target classification
* Add performance benchmark
* Optimize training time

1.0

* Support serializing/deserializing trees
* Implement Extratrees
