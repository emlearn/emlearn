
import numpy
from sklearn.base import BaseEstimator, TransformerMixin

class Quantizer(BaseEstimator, TransformerMixin):

    """
    Scales the features to fit a target range, usually a signed integer.
    Quantization is applied uniformly to all features.
    Scaling done using a linear multiplication, without any offset.

    If different features have very different scales,
    it is recommended to use a feature standardization transformation before this step.
    Examples: StandardScaler, RobustScaler, MinMaxScaler

    If the feature values have a very large range, or the range is very assymetrical,
    a non-linear pre-transformation may also be useful.
    """

    def __init__(self,
            max_quantile=0.0001,
            max_value=None,
            out_max=None, # automatically from dtype
            dtype='int16'):
        self.max_quantile = max_quantile
        self.max_value = max_value
        self.dtype = numpy.dtype(dtype)
        self.out_max = out_max

    def _get_out_max(self):
        if self.out_max is not None:
            return self.out_max

        info = None
        try:
            info = numpy.iinfo(self.dtype)
        except ValueError:
            info = numpy.finfo(self.dtype)
        if info is None:
            raise ValueError(f"Unsupported dtype {self.dtype}")

        out_max = info.max
        return out_max

    def fit(self, X, y=None):
        # TODO: normalize and check X,y using sklearn helpers
        out_max = self._get_out_max()

        if self.max_value is None:
            # learn the value from data
            high = 1.0-self.max_quantile
            low = self.max_quantile
            min_value = numpy.quantile(X, q=low, axis=None)
            max_value = numpy.quantile(X, q=high, axis=None)
            largest = max(max_value, -min_value)
            #print('mm', min_value, max_value, largest)
        else:
            largest = self.max_value            
    
        self.scale_ = out_max / largest
        #print('ss', self.scale_, out_max)
        return self

    def transform(self, X, y=None):

        # scale
        out = X * self.scale_

        # clip out-of-range values
        out_max = self._get_out_max()
        out = numpy.clip(out, -out_max, out_max)

        # quantize / convert dtype
        out = out.astype(self.dtype)

        # check post-conditions
        assert out.shape == X.shape
        assert out.dtype == self.dtype

        if y is None:
            return out

        return out, y
        
    def inverse_transform(self, X, y=None):

        # ensure workig with floats, not fixed-size integers
        out = X.astype(float)

        # clip out-of-range values
        out_max = self._get_out_max()
        out = numpy.clip(out, -out_max, out_max)

        # apply scale
        out = out / self.scale_

        assert out.shape == X.shape

        if y is None:
            return out

        return out, y

