
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

- Time-domain features. Root Mean Square (RMS) energy, Zero Crossing Rate (ZCR)
- Statistical summaries. min/max, mean/variance, kurtosis/skew et.c.
- Frequency domain (spectrum). Using Fourier transform (FFT), or filterbanks
- Time-frequency domain features (spectrogram). Using Short-Term Fourier Transform (STFT)

emlearn provides some tools for some of these,
and some guidelines for how to integrated custom feature extraction.

.. TODO: add some linkable reference documentation C code
.. TODO: link to each function documentation from the reference


Time-domain features
===========================

Root Mean Square (RMS) energy
Zero Crossing Rate (ZCR)

.. TODO: link an example

Digital filters
===========================

`eml_iir.h <https://github.com/emlearn/emlearn/blob/master/emlearn/eml_iir.h>`_

.. TODO: link an example

Spectrum
===========================

Fast Fourier Transform (FFT)

`eml_fft.h <https://github.com/emlearn/emlearn/blob/master/emlearn/eml_fft.h>`_

Spectrogram
===========================

Short-Time Fourier Transform (STFT)

Mel-frequency spectrogram

`eml_audio.h <https://github.com/emlearn/emlearn/blob/master/emlearn/eml_audio.h>`_,

.. TODO: link an example for mel-frequency spectrogram

Integrating feature extraction
===========================

.. TODO: explain how to make custom scikit-learn Transformer


.. TODO: add section on statistical summaries

