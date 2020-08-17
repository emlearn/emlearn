
import numpy
import numpy.fft

import pytest
import numpy.testing

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import librosa
import librosa.display
import pandas

import emlearn
import eml_audio

FFT_SIZES = [
    64,
    128,
    256,
    512,
    1024,
]
@pytest.mark.parametrize('n_fft', FFT_SIZES)
def test_rfft_simple(n_fft):
    signal = numpy.arange(0, n_fft)

    ref = numpy.fft.fft(signal, n=n_fft).real
    out = eml_audio.rfft(signal)
    diff = (out - ref)

    numpy.testing.assert_allclose(out, ref, rtol=1e-4)

def test_rfft_not_power2_length():
    with pytest.raises(Exception) as e:
        eml_audio.rfft(numpy.array([0,1,3,4,5]))

def fft_freqs(sr, n_fft):
    return numpy.linspace(0, float(sr)/2, int(1 + n_fft//2), endpoint=True)

def fft_freq(sr, n_fft, n):
    end = float(sr)/2
    steps = int(1 + n_fft//2) - 1
    return n*end/steps

def fft_freqs2(sr, n_fft):
    steps = int(1 + n_fft//2)
    return numpy.array([ fft_freq(sr, n_fft, n) for n in range(steps) ])

# Based on librosa
def melfilter(frames, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=True, norm=None):
    np = numpy
    if fmax is None:
        fmax = float(sr) / 2
    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_freqs(sr=sr, n_fft=n_fft)
    fftfreqs2 = fft_freqs2(sr=sr, n_fft=n_fft)
    assert fftfreqs.shape == fftfreqs2.shape
    numpy.testing.assert_almost_equal(fftfreqs, fftfreqs2)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    #ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        rlow = mel_f[i] - fftfreqs
        rupper = mel_f[i+2] - fftfreqs

        lower = -rlow / fdiff[i]
        upper = rupper / fdiff[i+1]

        # .. then intersect them with each other and zero
        w = np.maximum(0, np.minimum(lower, upper))
        if i == 4:
            print('wei', i, w[10:40])
        weights[i] = w

    refweighs = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
    numpy.testing.assert_allclose(weights, refweighs)

    return numpy.dot(frames, weights.T)
    #return numpy.dot(weights, frames)


# Basis for our C implementation,
# a mix of https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# and librosa
def melfilter_ref(pow_frames, sr, n_mels, n_fft):
    NFFT=n_fft
    sample_rate=sr
    nfilt=n_mels

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    fftfreqs = fft_freqs2(sr=sr, n_fft=n_fft)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        i = m-1
        fdifflow = hz_points[m] - hz_points[m - 1]
        fdiffupper = hz_points[m + 1] - hz_points[m]

        # TODO: fix/check divergence with librosa, who seems to compute
        # the peak filter value twice and select the lowest
        # sometimes the one below can give over 1.0 results at the center,
        # hence the clamp to 1.0, which does not seem right
        for k in range(f_m_minus, f_m+1):
            ramp = hz_points[i] - fftfreqs[k]
            w = -ramp / fdifflow
            w = max(min(w, 1), 0)
            fbank[i, k] = w
        for k in range(f_m, f_m_plus):
            ramp = hz_points[i+2] - fftfreqs[k+1]
            w = ramp / fdiffupper
            w = max(min(w, 1), 0)
            fbank[i, k+1] = w
        if i == 4:
            print('f', i, fbank[i][10:40])

    refweighs = librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=22050//2, htk=True, norm=None)
    #numpy.testing.assert_allclose(fbank, refweighs)

    filtered = numpy.dot(pow_frames, fbank.T)
    return filtered

def test_melfilter_basic():
    n_mels = 16
    n_fft = 512
    length = 1 + n_fft//2
    sr = 22050
    fmin = 0
    fmax = sr//2

    #input = numpy.ones(shape=length)
    input = numpy.random.rand(length)
    out = eml_audio.melfilter(input, sr, n_fft, n_mels, fmin, fmax)
    ref = librosa.feature.melspectrogram(S=input, htk=True, norm=None, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    ref2 = melfilter(input, sr, n_mels=n_mels, n_fft=n_fft)

    numpy.testing.assert_allclose(ref2, ref, rtol=1e-5)
    ref3 = melfilter_ref(input, sr, n_mels, n_fft)
    numpy.testing.assert_allclose(ref3, ref, rtol=1e-3)

    diff = out - ref

    fig, (ref_ax, out_ax, diff_ax) = plt.subplots(3)
    pandas.Series(out).plot(ax=out_ax)
    pandas.Series(ref).plot(ax=ref_ax)
    pandas.Series(diff).plot(ax=diff_ax)
    fig.savefig('melfilter.basic.png')

    assert ref.shape == out.shape
    numpy.testing.assert_allclose(out, ref, rtol=1e-3)


def test_melfilter_librosa():
    filename = librosa.util.example_audio_file()
    y, sr = librosa.load(filename, offset=1.0, duration=0.3)
    n_fft = 1024
    hop_length = 256
    fmin = 500
    fmax = 5000
    n_mels = 16

    spec = numpy.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    spec1 = spec[:,0]

    ref = librosa.feature.melspectrogram(S=spec1, sr=sr, norm=None, htk=True, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    out = eml_audio.melfilter(spec1, sr, n_fft, n_mels, fmin, fmax)

    fig, (ref_ax, out_ax) = plt.subplots(2)
    def specshow(d, ax):
        s = librosa.amplitude_to_db(d, ref=numpy.max)
        librosa.display.specshow(s, ax=ax, x_axis='time')
    specshow(ref.reshape(-1, 1), ax=ref_ax)
    specshow(out.reshape(-1, 1), ax=out_ax)
    fig.savefig('melfilter.librosa.png')

    assert ref.shape == out.shape
    numpy.testing.assert_allclose(ref, out, rtol=0.01)


@pytest.mark.skip('broken')
def test_melspectrogram():

    filename = librosa.util.example_audio_file()

    y, sr = librosa.load(filename, offset=1.0, duration=0.3)

    n_mels = 64
    n_fft = 1024
    fmin = 500
    fmax = 5000
    hop_size = n_fft

    # Only do one frame
    y = y[0:n_fft]
    ref = librosa.feature.melspectrogram(y, sr, norm=None, htk=True,
                                         fmin=fmin, fmax=fmax, n_fft=n_fft, n_mels=n_mels, hop_length=hop_size)
    out = eml_audio.melspectrogram(y, sr, n_fft, n_mels, fmin, fmax)
    ref = ref[:,0:1]
    out = out.reshape(-1,1)

    #out = melspec(y, sr, n_fft, n_mels, fmin, fmax, hop_length=hop_size)[:,:10]

    print('r', ref.shape)

    assert out.shape == ref.shape

    fig, (ref_ax, out_ax) = plt.subplots(2)
    def specshow(d, ax):
        s = librosa.amplitude_to_db(d, ref=numpy.max)
        librosa.display.specshow(s, ax=ax, x_axis='time')
        #librosa.display.specshow(s, ax=ax, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
    specshow(ref, ax=ref_ax)
    specshow(out, ax=out_ax)

    fig.savefig('melspec.png')

    print('mean', numpy.mean(ref), numpy.mean(out))
    print('std', numpy.std(ref), numpy.std(out))
    s = numpy.mean(ref) / numpy.mean(out)
    print('scale', s)
    out = out * s

    #print(out-ref)
    numpy.testing.assert_allclose(out, ref, rtol=1e-6);


def test_sparse_filterbank_ref():
    # testcase based on working example in STM32AI Function pack, mel_filters_lut_30.c
    mel = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=30, htk=False)
    sparse = emlearn.signal.sparse_filterbank(mel)

    expected_starts = [1, 7, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 77, 85, 93,
                       103, 114, 126, 139, 154, 170, 188, 208, 230, 254, 281, 311, 343, 379, 419]
    expected_ends = [12, 18, 24, 31, 37, 43, 49, 56, 62, 68, 76, 84, 92, 102, 113,
                     125, 138, 153, 169, 187, 207, 229, 253, 280, 310, 342, 378, 418, 463, 511]

    starts, ends, coeffs = sparse
    assert starts == expected_starts
    assert ends == expected_ends
    assert len(coeffs) == 968
    assert coeffs[0] == pytest.approx(1.6503363149e-03)
    assert coeffs[-1] == pytest.approx(2.8125530662e-05)


def test_sparse_filterbank_apply():
    n_fft = 1024
    n_mels = 30
    mel_basis = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels, htk=False)

    sparse = emlearn.signal.sparse_filterbank(mel_basis)
    starts, ends, coeffs = sparse
    assert len(starts) == n_mels
    assert len(ends) == n_mels

    name = 'fofofo'
    c = emlearn.signal.sparse_filterbank_serialize(sparse, name=name)

    assert name+'_lut' in c
    assert name+'_ends' in c
    assert name+'_starts' in c
    assert 'static const float' in c
    assert 'static const int' in c
    assert str(n_mels) in c

    data = numpy.ones(shape=mel_basis.shape[1]) * 100
    ref = numpy.dot(mel_basis, data)
    py_out = emlearn.signal.sparse_filterbank_reduce(sparse, data)
    numpy.testing.assert_allclose(py_out, ref, rtol=1e-5)

    c_out = eml_audio.sparse_filterbank(data, starts, ends, coeffs)
    numpy.testing.assert_allclose(c_out, ref, rtol=1e-5)
