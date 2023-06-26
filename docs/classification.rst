
.. Places parent toc into the sidebar
:parenttoc: True

.. _classification:

=========================
Classification
=========================

.. currentmodule:: emlearn


.. FIXME: write introduction

Applications
===========================
Detecting events using Machine Learning has a wide range of applications,
and it is commonly performed on sensor data using embedded systems.

.. table:: Application examples
    :widths: auto

    ============    =============================                       ============
      Area          Task                                                Sensor 
    ============    =============================                       ============
    Health          Detection of heart rythm irreguarity                Electrocardiogram (ECG)
    Wearables       Scene classification for contextual awareness       Sound
    Farming         Cattle behavior classification for health tracking  Accelerometer
    Robotics        Material identification for grippers                Capacitive
    Buildings       Human presence detection                            Radar
    ============    =============================                       ============


.. Classification of behaviors of free-ranging cattle using accelerometry signatures collected by virtual fence collars https://www.frontiersin.org/articles/10.3389/fanim.2023.1083272/full
.. Classification of Cattle Behaviours Using Neck-Mounted Accelerometer-Equipped Collars and Convolutional Neural Networks https://www.mdpi.com/1424-8220/21/12/4050                  
.. Eating monitoring



Classification models
===========================

.. FIXME: add list of models. CSV table?

A basic example of some classifier models can be found in
:ref:`sphx_glr_auto_examples_classifiers.py`.

Related
====================

Classification is an important component of :doc:`event_detection` systems.

