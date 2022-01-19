
"""
eml-mel-filterbank: Generate C code for a mel filterbank

Can be used with eml_sparse_filterbank()
"""

import argparse
import textwrap

import emlearn.signal


def mel_filterbank(args, name):
    import librosa

    norm = 'slaney' if args.normalize else None
    mel_basis = librosa.filters.mel(sr=args.samplerate, n_fft=args.fft, n_mels=args.bands,
                                    fmin=args.fmin, fmax=args.fmax, htk=args.htk, norm=norm)
    fmax = args.fmax if args.fmax is not None else args.samplerate/2
    frequencies = librosa.filters.mel_frequencies(n_mels=args.bands,
                                    fmin=args.fmin, fmax=fmax, htk=args.htk)

    sparse = emlearn.signal.sparse_filterbank(mel_basis)
    gen = emlearn.signal.sparse_filterbank_serialize(sparse, name=name, frequencies=frequencies,
            n_fft=args.fft, sr=args.samplerate, fmin=args.fmin, fmax=fmax)

    w = textwrap.wrap(gen, args.linewrap, replace_whitespace=False)
    wrapped = '\n'.join(w)
    return wrapped


def parse(args=None):
    parser = argparse.ArgumentParser(description='Generate filterbank for Mel spectrogram')
    a = parser.add_argument

    a('--bands', type=int, default=32,
      help='Number of mel bands')
    a('--fft', type=int, default=1024,
      help='Number of coefficients in FFT')
    a('--samplerate', type=int, default=16000,
      help='Samplerate used')

    a('--fmin', type=int, default=0,
      help='Frequency of lowest band. Default: 0')
    a('--fmax', type=int, default=None,
      help='Frequency of highest band. Default: samplerate/2')
    a('--htk', type=bool, default=False,
      help='Use HTK style filter spacing')
    a('--normalize', type=bool, default=True,
      help='Normalize to have constant energy per band')

    a('--name', type=str, default='',
      help='Output file. Default: ')
    a('--out', type=str, default='',
      help='Output file. Default: $name.h')
    a('--linewrap', type=int, default=70,
      help='Maximum width of lines')

    parsed = parser.parse_args(args)
    return parsed


def main():
    args = parse()
    out = args.out
    name = args.name
    if not name:
        name = '_'.join(['mel', str(args.fft), str(args.bands)])
    if not out:
        out = name+'.h'

    preamble = '// This file was generated with emlearn using eml-mel-filterbank\n\n'
    gen = mel_filterbank(args, name)
    wrapped = preamble + gen + '\n\n'

    with open(out, 'w') as f:
        f.write(wrapped)
    print('Wrote to', out)


if __name__ == '__main__':
    main()
