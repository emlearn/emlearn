
.. Places parent toc into the sidebar
:parenttoc: True

.. _anomaly_detection:

=========================
Anomaly Detection
=========================

.. currentmodule:: emlearn

.. FIXME: finish introduction

Anomaly Detection is the task of identifying samples that are *abnormal* / *anomalous*,
meaning that they deviate considerably from some *normal* / *typical* data.
The task is highly related to **Outlier Detection** and **Novelty Detection**. 
Many of the tools and techniques mentioned here can be transferred directly to those tasks.

Anomaly Detection is normally approached as an unsupervised learning problem,
with the training data consisting only of normal data (potentially with a low degree of contamination).
Any deviation from this normal training data is to be considered an *anomaly*.
This is sometimes called *one-class classification*, as it focuses on modelling a single class (the normal class).

The one-class approach ensures that novel anomalies (different from those seen during development),
are correctly classified as anomalies.
Using a supervised binary classification approach usually has the problem that novel datapoints
may be incorrectly marked as normal, because it falls inside a decision boundary in training set.

While labled data is not used in the training set,
in practice a small labeled dataset is critical for performance evaluation.


Applications
===========================
Anomaly Detection has wide range of applications
using sensor data and embedded systems.
Here are a few examples.

.. table:: Application examples
    :widths: auto

    ============    =================================================                       ============
      Area          Task                                                                    Sensor 
    ============    =================================================                       ============
    Industrial      Condition Monitoring of rotating machninery                             Accelerometer
    Industrial      Detecting fault in machines                                             Microphone
    Electronics     Detecting issues in Lithium Ion batteries                               Electrical/thermal
    Automotive      Instrusion and fault detection in CANBus networks                       CANBus
    Robotics        Monitoring executed tasks for faults                                    Mix
    Health          Detection of anomalous heartbeats                                       Electrocardiogram (ECG)      
    ============    =================================================                       ============


.. TODO: link to section on class imbalance. Maybe split out from Event Detection

.. TODO: describe setting anomaly score thresholds

Performance evaluation
===========================



Anomaly Detection models
===========================

emlearn supports a selection of Anomaly Detection models.

.. table:: Supported models for Anomaly Detection
   :widths: auto
   
   ============================     ======
   Algorithm                        Implementation
   ============================     ======
   Gaussian Mixture Model (GMM)     `GaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html>`_, `BayesianGaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html>`_
   Mahalanobis distance             `EllipticEnvelope <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html>`_
   ============================     ======

A basic example of some of the models can be found in
:ref:`sphx_glr_auto_examples_anomaly_detection.py`.


Outlier detection for handling unknown data
===========================

Anomaly/outlier detection models are used in :doc:`_classification` or :doc:`_regression` systems,
in order to handle input data that are outside the data distribution of a trained model.
This is to prevent spurious results on out-of-distribution inputs.
The input data of the classifier or regressor is also passed through an outlier detection model,
and the outliers are classified as "unknown".


.. TODO: write section on combined systems
.. Anomaly Detection models are often components in monitoring systems.
.. They may also be combined with classification or regression
.. to do fault diagnostics, fault localization et.c.

