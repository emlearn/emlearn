
import numpy
import numpy.fft

import pytest
import numpy.testing

import emlearn
import eml_audio

def test_rfft_simple():
    ss = numpy.arange(0, 1024)
    ref = numpy.fft.fft(ss, n=1024).real
    out = eml_audio.rfft(ss)

    numpy.testing.assert_allclose(out, ref, rtol=1e-5)

def test_rfft_not_power2_length():
    with pytest.raises(Exception) as e:
        eml_audio.rfft(numpy.array([1,3,4,5]))

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


