
from scipy import signal
import numpy
import pandas

import pytest

import eml_signal


def plot_freq_response(noise, a, b, fs=44100):

    def spec(sig):
        return signal.periodogram(sig, fs=fs, window='hann', scaling='spectrum')

    f, noise_s = spec(noise)
    print('noise', noise_s.shape)

    df = pandas.DataFrame({
        'f': f,
        'original': noise_s,
        'a': spec(a)[1],
        'b': spec(b)[1],
    })
    df.plot(x='f', logy=True, logx=True, xlim=(10, fs/2))


def noisy_chirp(fs=44100, seconds=1.0):
    t = numpy.linspace(0, 1, int(seconds*fs), False)
    chirp = signal.chirp(t, f0=6, f1=fs/2.0, t1=seconds, method='linear')
    noise = numpy.random.normal(0, 1.0, size=len(t))
    sig = chirp + 0.05*noise
    return t, sig


FS = 32000
FILTERS = {
    'cheby1-order1-lowpass': signal.cheby1(N=1, rp=3, Wn=1000, btype='lp', fs=FS, output='sos'),
    'cheby1-order2-highpass': signal.cheby1(N=2, rp=3, Wn=2000, btype='hp', fs=FS, output='sos'),
    'cheby1-order4-highpass': signal.cheby1(N=4, rp=3, Wn=2000, btype='hp', fs=FS, output='sos'),
    'cheby1-order5-bandpass': signal.cheby1(N=5, rp=12, Wn=[2000, 4000], btype='bp', fs=FS, output='sos'),
    'cheby1-order12-highpass': signal.cheby1(N=12, rp=12, Wn=4000, btype='hp', fs=FS, output='sos'),
}


@pytest.mark.parametrize('filtername', FILTERS.keys())
def test_iir_filter(filtername):
    sos = FILTERS[filtername]
    assert len(sos.shape) == 2
    assert sos.shape[1] == 6, sos.shape

    t, noise = noisy_chirp(fs=FS, seconds=1.0)

    ref = signal.sosfilt(sos, noise)
    out = eml_signal.iirfilter(sos, noise)
    rel_err = numpy.abs((out-ref)/ref)

    assert ref.shape == out.shape, out.shape
    numpy.testing.assert_allclose(ref, out, atol=1e-4)
    assert numpy.median(rel_err) < 1e-4