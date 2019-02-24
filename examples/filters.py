
import numpy
import scipy.signal
from matplotlib import pyplot as plt

import acoustics
import librosa

def bandpass_filter(lowcut, highcut, fs, order, output='sos'):
    assert order % 2 == 0, 'order must be multiple of 2'
    assert highcut*0.95 < (fs/2.0), 'highcut {} above Nyquist for fs={}'.format(highcut, fs)
    assert lowcut > 0.0, 'lowcut must be above 0'

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    output = scipy.signal.butter(order / 2, [low, high], btype='band', output=output)
    return output

def filterbank(center, fraction, fs, order):
    reference = acoustics.octave.REFERENCE

    # remove bands above Nyquist
    center = [ f for f in center if f < fs/2.0 ]
    center = numpy.asarray(center)
    indices = acoustics.octave.index_of_frequency(center, fraction=fraction, ref=reference)

    # use the exact frequencies for the filters
    center = acoustics.octave.exact_center_frequency(None, fraction=fraction, n=indices, ref=reference)  
    lower = acoustics.octave.lower_frequency(center, fraction=fraction)
    upper = acoustics.octave.upper_frequency(center, fraction=fraction)

    nominal = acoustics.octave.nominal_center_frequency(None, fraction, indices)

    # XXX: use low/highpass on edges?
    def f(low, high):
        return bandpass_filter(low, high, fs=fs, order=order)
    filterbank = [ f(low, high) for low, high in zip(lower, upper) ]

    return nominal, filterbank

def octave_filterbank(fs, order=8):
    from acoustics.standards import iec_61672_1_2013 as iec_61672
    center = iec_61672.NOMINAL_OCTAVE_CENTER_FREQUENCIES

    return filterbank(center, fraction=1, fs=fs, order=order)

def third_octave_filterbank(fs, order=8):
    from acoustics.standards import iec_61672_1_2013 as iec_61672
    center = iec_61672.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

    return filterbank(center, fraction=3, fs=fs, order=order)


def plot_response(filterbank, fs):

    fig, ax = plt.subplots(1)

    for center, sos in zip(filterbank[0], filterbank[1]):
        w, h = scipy.signal.sosfreqz(sos, worN=4096, fs=fs)
        db = 20*numpy.log10(numpy.abs(h))
        ax.plot(w, db)
    ax.set_ylim(-60, 5)
    ax.set_xlim(20.0, 20e3) 
    #ax.set_xscale('log')

    return fig

def main():
    fs = 44100

    third  = third_octave_filterbank(fs)
    whole = octave_filterbank(fs)

    # TODO: extract time-frequency features with these filterbanks
    # compare with mel-spectrograms
    # use for classification

    print('1/3 oct', len(third[0]), third[0])
    print('1/1 oct', len(whole[0]), whole[0])

    f = plot_response(whole, fs=fs)
    f.savefig('whole.png')

    f = plot_response(third, fs=fs)
    f.savefig('third.png')


if __name__ == '__main__':
    main()

