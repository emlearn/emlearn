
"""eml-window-function: Generating C code for window functions

Part of the emlearn project: https://emlearn.org
Redistributable under the MIT license
"""

import argparse
import textwrap

from .. import cgen

# Supports everything without parameters in scipy.signal.get_window
_known = 'boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann'
known_window_types = tuple(_known.split(', '))


def parse(args=None):
    parser = argparse.ArgumentParser(description='Generate lookup table for window functions')
    a = parser.add_argument

    a('--window', type=str, default='hann',
      help='Window function to use. Supported: \n' + '|'.join(known_window_types))
    a('--length', type=int, default=1024,
      help='Number of coefficients in window')
    a('--symmetric', default=False, action='store_true',
      help='Whether to use a symmetric window. Defaults to False, normal for FFT')
    a('--name', type=str, default='',
      help='Name of the generate C array')
    a('--out', type=str, default='',
      help='Output file. Default: $name.h')
    a('--linewrap', type=int, default=70,
      help='Maximum width of lines')

    parsed = parser.parse_args(args)
    return parsed


def window_function(name, window_type, length, fft_mode, linewrap):
    import scipy.signal

    window = scipy.signal.get_window(window_type, length, fftbins=fft_mode)
    arrays = [
        cgen.array_declare(name, length, values=window),
        cgen.constant_declare(name+'_length', val=length),
    ]
    gen = '\n'.join(arrays)

    w = textwrap.wrap(gen, linewrap)
    wrapped = '\n'.join(w)
    return wrapped


def main():
    args = parse()

    window_type = args.window
    length = args.length
    fft_mode = not args.symmetric
    name = args.name
    out = args.out
    if not name:
        name = '_'.join([window_type, str(length), 'lut'])
    if not out:
        out = name+'.h'

    if window_type not in known_window_types:
        print('Warning: Unknown window type {}. Known:\n {}'.format(window_type, known_window_types))

    preamble = '// This file was generated with emlearn using eml-window-function\n\n'

    wrapped = window_function(name, window_type, length, fft_mode, args.linewrap)
    wrapped = preamble + wrapped

    with open(out, 'w') as f:
        f.write(wrapped)
    print('Wrote to', out)


if __name__ == '__main__':
    main()
