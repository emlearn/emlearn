
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
    f = Quantizer(dtype=dtype, max_value=10.0)
    f.fit(a)
    out = f.transform(a)
    assert out.dtype == numpy.dtype(dtype)
    oo = f.inverse_transform(out)
    expected_decimals_correct = 3
    if '8' in dtype:
        expected_decimals_correct = 1
    assert_almost_equal(a, oo, decimal=expected_decimals_correct)


