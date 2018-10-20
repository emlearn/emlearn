
## General

0.3

* Include emnet
* Include embayes
* Standalone example on microcontroller. XOR?
* Use absolute path by default for library imports in generated code

0.4

* Consolidate error handling. Use EmlError, EML_PRE/POSTCONDITION, EML_CHECK_ERROR from `emnet`
* Implement some iterators like EML_APPLY(vector, 0, vector->length, func) and EML_MAP_INTO(in, out, 0, out->length, func);

## audio

0.3

* Add tests for melspectrogram
* Unhardcode FFT length
* Add some pipeline/processor.
Short audio buffers in, feature extraction, normalization, classifier, predictions out.

## trees

0.3

* Use floats by default

1.0

* Support returning probabilities
* Support serializing/deserializing trees

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

