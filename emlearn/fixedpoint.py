
"""
Tools for working with fixed-point numbers in Python
"""

import dataclasses
import numpy

@dataclasses.dataclass
class FixedPointFormat:
    integer_bits: int
    fraction_bits: int
    sign_bits: int = 1
    
    @property
    def total_bits(self):
        t = self.integer_bits + self.fraction_bits + self.sign_bits
        return t

    @property
    def ctype(self):
        assert self.total_bits == 32, 'Only 32 bit widths supported at the moment'
        types = {
            (15, 16, 1): 'eml_q16_t',
        }
        spec = (self.integer_bits, self.fraction_bits, self.sign_bits)
        ctype = types.get(spec, None)
        if ctype is None:
            return 'eml_fixed32_t'    
        else:
            return ctype


def from_float(numbers : numpy.array, fmt : FixedPointFormat):
    """
    Convert numbers (floats) to fixed-point
    """
    a = numbers.astype(float)
    out = (a * (1 << fmt.fraction_bits)).astype(int)
    return out

def to_float(fixed: numpy.array, fmt : FixedPointFormat):
    """
    Convert fixed-point numbers to floats
    """
    a = fixed.astype(float)
    out = a / (1 << fmt.fraction_bits)
    return out
