
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

if __name__ == '__main__':

    sos = create_lowpass(samplerate=50)

    out = 'build/gravity_lowpass.npy'
    flat = sos.flatten()
    numpy.save(out, sos)
    #print(sos.shape, sos)
    print(flat.shape, flat)


    name = 'gravity_lowpass'
    length = len(flat)
    code = '// Generated using lowpass.py\n\n' + \
        f'#define {name}_length {length}\n' + \
        emlearn.cgen.array_declare(name=f'{name}_values', values=flat) + '\n'

    header = 'build/gravity_filter.h'

    with open(header, 'w') as f:
        f.write(code)

    print('Wrote', header)

