
## General

0.3

* Standalone example on microcontroller. XOR?

0.4

* Setup documentation build (Sphinx?) 
* Consolidate error handling. Use EmlError, EML_PRE/POSTCONDITION, EML_CHECK_ERROR everywhere

Later

* Implement some iterators like EML_APPLY(vector, 0, vector->length, func) and EML_MAP_INTO(in, out, 0, out->length, func);

## audio

0.3

* Add tests for melspectrogram
* Unhardcode FFT length
* Add some pipeline/processor.
Short audio buffers in, feature extraction, normalization, classifier, predictions out.

0.4

* Add an example. Wakeword detection? Keyword classification?
* Implement low-level features.
RMS, zero-crossings, spectral centroid, spectral flatness


## nets

- Add support for CNNs. 1D+2D convolutions, pooling (Keras).
- Add support for RNNs. SimpleRNN, LSTM, GRU
- Support quantized models (8 bits)
- Supports strides and dilation in convolution layers

## trees

0.3

* Use floats by default

Probably

* Support sklearn GradientBoostingClassifier
* Support regression trees
* Support weighted voting
* Implement Isolation Forests (requires path/depths)

Maybe

* Support [XGBoost](https://github.com/dmlc/xgboost) learning of trees
* Support [LightGBM](https://github.com/Microsoft/LightGBM) learning of trees
* Support [CatBoost](https://github.com/catboost/catboost) learning of trees
* Support/Implement a Very Fast Decision Tree (VFDT) learning algorithm

