
import emlearn

import pickle

# Load from file
with open('combined.estimator.pickle', 'rb') as f:
    estimator = pickle.load(f)


    print(estimator)

    c_model = emlearn.convert(estimator,
        dtype='float',
        method='inline',
        leaf_bits=4,
    )

    out = 'build/motion_model.h'
    c_model.save(file=out, name='motion_model')

    print('Wrote', out)
