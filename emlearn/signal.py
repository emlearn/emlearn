
from . import cgen

import numpy


def sparse_filterbank(mels):
    starts = []
    ends = []
    coeffs = []
    for mel_idx in range(mels.shape[0]):
        mel = mels[mel_idx]
        nonzero = numpy.nonzero(mel)[0]
        first, last = nonzero[0], nonzero[-1]
        starts.append(first)
        ends.append(last)
        coeffs += list(mel[nonzero])

    return starts, ends, coeffs


def sparse_filterbank_serialize(sparse, name, frequencies=None, n_fft=None, sr=None, fmin=None, fmax=None):
    starts, ends, coeffs = sparse
    assert len(starts) == len(ends)
    n_bands = len(starts)

    arrays = [
        cgen.constant_declare(name+'_bands', val=n_bands),
        cgen.array_declare(name+'_starts', len(starts), dtype='int', values=starts),
        cgen.array_declare(name+'_ends', len(ends), dtype='int', values=ends),
        cgen.constant_declare(name+'_length', val=len(coeffs)),
        cgen.array_declare(name+'_lut', len(coeffs), values=coeffs),
    ]
    if frequencies is not None:
        arrays += [
            cgen.array_declare(name+'_frequencies', len(frequencies), dtype='float', values=frequencies),
        ]
    if n_fft is not None:
        arrays += [
            cgen.constant_declare(name+'_nfft', val=n_fft),
        ]
    if sr is not None:
        arrays += [
            cgen.constant_declare(name+'_samplerate', val=sr),
        ]
    if fmax is not None:
        arrays += [
            cgen.constant_declare(name+'_fmax', val=fmax),
        ]
    if fmin is not None:
        arrays += [
            cgen.constant_declare(name+'_fmin', val=fmin),
        ]

    out = '\n\n'.join(arrays)
    return out


def sparse_filterbank_reduce(sparse, input):
    starts, ends, coeffs = sparse
    assert len(starts) == len(ends)

    offset = 0
    out = numpy.zeros(shape=(len(starts),))
    for i in range(len(starts)):
        for j in range(starts[i], ends[i]+1):
            out[i] += input[j] * coeffs[offset]
            offset += 1

    return out
