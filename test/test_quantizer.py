
from emlearn.preprocessing import Quantizer

import pytest
import numpy
from numpy.testing import assert_almost_equal

SUPPORTED_DTYPES=['int16', 'int8', 'float16', 'i4']

# TODO: test setting quantile
# TODO: test defaults
# TODO: test setting max_value
# TODO: test usage as part of a scikit-learn Pipeline

@pytest.mark.parametrize('dtype', SUPPORTED_DTYPES)
def test_feature_quantizer_simple(dtype):

    rng = numpy.random.default_rng()
    a = rng.normal(size=(10, 3))

    # round-tripping data is approximately equal
    f = Quantizer(dtype=dtype, max_value=9.0)
    f.fit(a)
    out = f.transform(a)
    assert out.dtype == numpy.dtype(dtype)
    oo = f.inverse_transform(out)
    expected_decimals_correct = 3
    if '8' in dtype:
        expected_decimals_correct = 1
    assert_almost_equal(a, oo, decimal=expected_decimals_correct)

def test_feature_quantizer_out_of_bounds_clipped():

    dtype = 'int16'
    a = numpy.array([[
        -100, -10, -1, -2, -3, 0, 1, 10, 1000,
    ]]).astype(float)

    max_value = 9
    f = Quantizer(dtype=dtype, max_value=max_value)
    f.fit(a)
    out = f.transform(a)
    oo = f.inverse_transform(out)

    # clipping works when roundtripped
    assert oo[0, 0] == -max_value
    assert oo[0, -1] == max_value
    expect_roundtripped = numpy.clip(a, -max_value, max_value)

    expected_decimals_correct = 3
    if '8' in dtype:
        expected_decimals_correct = 1
    assert_almost_equal(expect_roundtripped, oo, decimal=expected_decimals_correct)


