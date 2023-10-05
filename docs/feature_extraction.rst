
.. Places parent toc into the sidebar
:parenttoc: True

.. _feature_extraction:

=========================
Feature extraction
=========================

.. currentmodule:: emlearn

The raw samples of the sensor data can be hard for a model to learn from,
as it may be high dimensional and have a low signal to noise ratio wrt to the task.

Therefore it can be useful to apply feature extraction to make the problem more tractable.
Common feature extraction techniques for time-series and sensor-data include:

- Time-domain features
- Statistical summaries
- Frequency domain (spectrum)
- Time-frequency domain (spectrogram)

emlearn provides some tools for some of these.

Time-domain features
===========================

Typical features are

- Root Mean Square (RMS) energy
- Zero Crossing Rate (ZCR)

.. TODO: refer to C code/functions for these
.. TODO: link an example

Statistical summaries
===========================

Statistical summaries are ways of extracting compact representations of sets of values.
This can be useful on time-series data, on spectrum data, or other high-dimensional signals.

Typical features used are:

- minimum/maximum
- peak2peak
- variance / standard deviation
- mean
- median
- 25/75 percentile, and Interquartile Distance

.. TODO: refer to C code/functions for these
.. TODO: link an example. Maybee HAR on accelerometer data

Digital filters
===========================

Digital filters can be very useful to process a time-series signal.

`Infinite Impulse Response (IIR) <https://en.wikipedia.org/wiki/Infinite_impulse_response>`_ filter is one way of creating digital filters.
These are useful for:

- Low-pass filters
- High-pass filters
- Band-pass filters

In Python the IIR filter coefficients can be designed with
`scipy.signal.iirfilter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html>`_ (by specifying filter order and critical frequencies)
or `scipy.signal.iirdesign <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html>`_ (by specifying stop/passband frequencies and gains).

The design can be output as second-order sections (format='sos'),
and then realized in using :doc:`eml_iir`.

.. MAYBE. Link papers on learning IIR filters
.. Ex: https://www.dafx.de/paper-archive/2020/proceedings/papers/DAFx2020_paper_52.pdf

.. TODO: link an example

Spectrum (frequency domain)
=====================================

Many phenomena can be easier to separate in the frequency domain, rather than the time-domain.
The most common way to transform is using the FFT,
which is implemented in :doc:`eml_fft`.

.. TODO: show an image of spectrum

Spectrogram (time-frequency domain)
========================================

A spectrogram decomposes a time-series into both time and frequency,
creating a 2d image-like representation.
Spectrograms are commonly used with a wide range of input data, such as:
sound, accelerometer, Electrocardiogram (ECG), seismology, etc.

It is most commonly done by applying the FFT to overlapped consecutive time windows.
This technique is called Short-Time Fourier Transform (STFT).
An alternative is to use multiple FIR or IIR bandpass filters to form a filterbank.

A special case of a spectrogram called the Mel-frequency spectrogram
is particularly popular for audio machine learning applications.

Code can be found in :doc:`eml_audio`.

.. TODO: show an image of spectrogram
.. TODO: link an example for mel-frequency spectrogram

Integrating feature extraction
===================================

It is practical to start prototyping and testing feature extraction approaches in Python,
using the wide range of available functions and libraries.
But once the appropriate feature extraction method has been identified,
it is normally implemented in C to run on the target device.
It is recommended to use the same C code also during training.
This reduces the risk of divergence in feature extraction between target and training,
which can have very negative impact on predictive performance.

There are two main approaches to use C code during training (in Python):

- Create a C program and use files to pass input/output data
- Create Python bindings for the C functions.

Python bindings can be created using `pybind11 <https://github.com/pybind/pybind11>`_
or `CFFI <https://cffi.readthedocs.io/en/latest/>`_.

.. TODO: explain how to make custom scikit-learn Transformer


