
.. Places parent toc into the sidebar
:parenttoc: True

.. _regression:

=========================
Regression
=========================

.. currentmodule:: emlearn

Regression is the task of predicting / estimating a *continuous* value.
Training is performed using supervised machine learning, using a labeled dataset.
It can be applied to standard a set of input values, or to time-series data.
The output can be either a single data point for a time-series, or one per time-step.


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
    Industrial      Remaining Useful Life (RUL) estimation              *
    Industrial      Prediction of failures using Time-to-Event          Accelerometer etc.
    ============    ==============================================      ============


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


Related
===========================

Some specialized task formulations that use regression are:
Time-series *forecasting*, for predicting future values,
and Time-to-Event estimation, which is similar to to :doc:`event_detection`.

