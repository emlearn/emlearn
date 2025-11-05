
import argparse

import pandas
import numpy
import emlearn
from scipy.signal import iirfilter, sosfiltfilt

def create_lowpass(samplerate, cutoff=0.5, order=4):
    nyquist = samplerate / 2
    normalized_cutoff = cutoff / nyquist
    
    # NOTE: in theory an Elliptic filter would allow sharper transition
    sos = iirfilter(order, normalized_cutoff, 
                   btype='lowpass', ftype='butter', output='sos')

    return sos

def parse():
    parser = argparse.ArgumentParser(description='Generate lowpass filter')
    parser.add_argument('--out', type=str, default='build/gravity_filter.h',
                       help='Output file path')
    parser.add_argument('--format', type=str, default='header', 
                       choices=['header', 'npy'],
                       help='Output format')
    parser.add_argument('--name', type=str, default='gravity_lowpass',
                       help='Variable name prefix')
    parser.add_argument('--samplerate', type=float, default=50,
                       help='Sample rate in Hz')
    parser.add_argument('--cutoff', type=float, default=0.5,
                       help='Cutoff frequency in Hz')
    parser.add_argument('--order', type=int, default=4,
                       help='Filter order')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    
    sos = create_lowpass(samplerate=args.samplerate, 
                        cutoff=args.cutoff, 
                        order=args.order)
    flat = sos.flatten()
    
    if args.format == 'npy':
        numpy.save(args.out, sos)
        print('Wrote', args.out)
    else:
        length = len(flat)
        code = '// Generated using lowpass.py\n\n' + \
            f'#define {args.name}_length {length}\n' + \
            emlearn.cgen.array_declare(name=f'{args.name}_values', values=flat) + '\n'
        
        with open(args.out, 'w') as f:
            f.write(code)
        print('Wrote', args.out)
