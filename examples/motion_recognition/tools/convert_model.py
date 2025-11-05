
"""Convert a scikit-learn RandomForest to a C header using emlearn"""

import emlearn
import pickle
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Convert model to C code')
    parser.add_argument('--model', type=str, default=None,
                       help='Input model pickle file')
    parser.add_argument('--out', type=str, default='build/motion_model.h',
                       help='Output header file path')
    parser.add_argument('--name', type=str, default='motion_model',
                       help='Model name in generated code')
    parser.add_argument('--dtype', type=str, default='float',
                       choices=['float', 'double', 'int8', 'int16'],
                       help='Data type for model parameters')
    parser.add_argument('--method', type=str, default='inline',
                       choices=['inline', 'loadable'],
                       help='Code generation method')
    parser.add_argument('--leaf_bits', type=int, default=4,
                       help='Number of bits for leaf values')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    
    # Load from file
    with open(args.model, 'rb') as f:
        estimator = pickle.load(f)
        print(estimator)
        
        c_model = emlearn.convert(estimator,
            dtype=args.dtype,
            method=args.method,
            leaf_bits=args.leaf_bits,
        )
        
        c_model.save(file=args.out, name=args.name)
        print('Wrote', args.out)
