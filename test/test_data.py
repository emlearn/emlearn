
import emlearn

import pytest
import numpy

def test_data_roundtrip_minimal():
    inp = numpy.int32([
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
    ])

    buf = emlearn.data.serialize(inp)
    assert len(buf) > 20
    out = emlearn.data.deserialize(buf)
    assert inp.shape == out.shape
    numpy.testing.assert_allclose(inp, out)

