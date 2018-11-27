
import numpy
import numpy.fft

import pytest
import numpy.testing

import emlearn
import eml_audio

FFT_SIZES = [
    64,
    128,
    256,
    512,
    1024,
    2048,
]
@pytest.mark.parametrize('n_fft', FFT_SIZES)
def test_rfft_simple(n_fft):
    signal = numpy.arange(0, n_fft)

    ref = numpy.fft.fft(signal, n=n_fft).real
    out = eml_audio.rfft(signal)
    diff = (out - ref)

    numpy.testing.assert_allclose(out, ref, rtol=1e-5)

def test_rfft_not_power2_length():
    with pytest.raises(Exception) as e:
        eml_audio.rfft(numpy.array([0,1,3,4,5]))

@pytest.mark.skip('wrong scaling?')
def test_melspectrogram():

    import librosa

    y, sr = librosa.load('data/ff1010bird/{}.wav'.format(19037), offset=0)

    ref = librosa.feature.melspectrogram(y, sr, fmin=0, fmax=None, n_fft=1024, n_mels=64, norm=None, htk=True)
    out = eml_audio.melspectrogram(y, sr, n_fft=1024, fmin=0, fmax=None, n_mels=64)
    ref = ref[:,1:-1]

    print(numpy.mean(ref))
    print(numpy.mean(out))

    print(out-ref)
    numpy.testing.assert_allclose(out, ref, rtol=1e-6);


