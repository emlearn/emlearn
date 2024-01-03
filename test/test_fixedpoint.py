
from emlearn import fixedpoint

import pytest
import numpy
from numpy.testing import assert_allclose


def test_fixedpoint_roundtrip():
    fmt = fixedpoint.FixedPointFormat(integer_bits=15, fraction_bits=16)

    rng = numpy.random.default_rng()
    numbers = rng.uniform(low=-(2.0**15), high=+(2.0**15), size=(12,))
    #numbers = numpy.array([1, 2, 0.5])
    fixed = fixedpoint.from_float(numbers, fmt=fmt)
    assert fixed.dtype == int
    roundtrip = fixedpoint.to_float(fixed, fmt=fmt)

    assert_allclose(numbers, roundtrip)

@pytest.mark.skip("Not implemented")
def test_fixedpoint_compat():
    # Test with respect to EML_Q16_FROMFLOAT / EML_Q16_TOFLOAT
    pass
