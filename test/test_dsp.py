
import eml_audio as eml_dsp

import pytest
import numpy


@pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
@pytest.mark.parametrize('kernel_size', [3, 5, 7])
def test_convolve1d(mode, kernel_size):
    input_size = kernel_size * 2
    stride = 1
    signal = numpy.random.random(input_size).astype(float)
    kernel = numpy.random.random(kernel_size).astype(float)

    ref = numpy.convolve(signal, kernel, mode=mode).astype(float)
    out = eml_dsp.convolve1d(signal, kernel, mode, stride)

    numpy.testing.assert_allclose(ref, out, atol=0.0001)
