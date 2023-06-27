
.. Places parent toc into the sidebar
:parenttoc: True

.. _regression:

=========================
Regression
=========================

.. currentmodule:: emlearn


.. FIXME: write introduction

Applications
===========================
Detecting events using Machine Learning has a wide range of applications,
and it is commonly performed on sensor data using embedded systems.

.. table:: Application examples
    :widths: auto

    ============    ==============================================      ============
    Area            Task                                                Sensor 
    ============    ==============================================      ============
    Electronics     Battery power estimation                            Voltage/current
    Robotics        Distance estimation                                 Ultrasound
    Health          Breathing rate estimation                           Sound
    Sensors         Calibration of air quality sensors                  PM2.5 sensor
    Industrial      Gas concentration estimation                        Metal-Oxide semiconductor (MOS)
    ============    ==============================================      ============


..     Traffic         Vehicle speed estimation


Classification models
===========================

.. table:: Supported regression models
   :widths: auto
   
   ============================     ======
   Algorithm                        Implementation
   ============================     ======
   RandomForest                     `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_
   ExtraTrees                       `ExtraTreesRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_
   DecisionTree                     `DecisionTreeRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
   Multi-Layer-Perceptron           `MLPRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_,  `keras.Sequential <https://keras.io/api/models/sequential/>`_
   ============================     ======


A basic example of some of the regressions models can be found in
:ref:`sphx_glr_auto_examples_regression.py`.


.. TODO: try to add links to related pages


