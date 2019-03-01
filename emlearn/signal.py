
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


def sparse_filterbank_serialize(sparse, name):
    starts, ends, coeffs = sparse

    arrays = [
        cgen.array_declare(name+'_starts', len(starts), dtype='int', values=starts),
        cgen.array_declare(name+'_ends', len(ends), dtype='int', values=ends),
        cgen.array_declare(name+'_lut', len(coeffs), values=coeffs),
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
