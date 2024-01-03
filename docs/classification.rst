
.. Places parent toc into the sidebar
:parenttoc: True

.. _classification:

=========================
Classification
=========================

.. currentmodule:: emlearn

Classification is the task of predicting /estimating a discrete category (the *class*) from input data.
Training is performed using supervised machine learning, using a labeled dataset.
Classification can be binary or multi-class.

Sometimes a single model has multiple outputs, multi-label classification.
This is not currently supported in emlearn.
Instead one classification model per output must be used.


Applications
===========================
Detecting events using Machine Learning has a wide range of applications,
and it is commonly performed on sensor data using embedded systems.

.. table:: Application examples
    :widths: auto

    ============    ==========================================================  ====================================
    Area            Task                                                        Sensor 
    ============    ==========================================================  ====================================
    Health          Detection of heart rythm irreguarity                        Electrocardiogram (ECG)
    Wearables       Scene classification for contextual awareness               Sound
    Farming         Cattle behavior classification for health tracking          Accelerometer
    Robotics        Material identification for grippers                        Capacitive
    Buildings       Human presence detection                                    Radar
    ============    ==========================================================  ====================================


.. Classification of behaviors of free-ranging cattle using accelerometry signatures collected by virtual fence collars https://www.frontiersin.org/articles/10.3389/fanim.2023.1083272/full
.. Classification of Cattle Behaviours Using Neck-Mounted Accelerometer-Equipped Collars and Convolutional Neural Networks https://www.mdpi.com/1424-8220/21/12/4050                  
.. Eating monitoring



Classification models
===========================

.. table:: Supported classification models
   :widths: auto
   
   ============================     ======
   Algorithm                        Implementation
   ============================     ======
   RandomForest                     `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
   ExtraTrees                       `ExtraTreesClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>`_
   DecisionTree                     `DecisionTreeClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_
   Multi-Layer-Perceptron           `MLPClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_,  `keras.Sequential <https://keras.io/api/models/sequential/>`_
   Gaussian Naive Bayes             `GaussianNB <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_
   Nearest Neighbors                `KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
   ============================     ======

A basic example showing some of these classifier models can be found in
:ref:`sphx_glr_auto_examples_classifiers.py`.

Related
====================

Classification is an important component of :doc:`event_detection` systems.

