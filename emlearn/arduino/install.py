
"""
Install emlearn as Arduino library
"""

from ..common import get_include_dir
from ..utils.fileutils import ensure_dir_single

import sys
import argparse
import os
import shutil
import glob

class Error(ValueError):
    pass


def install_arduino_library(emlearn_include_dir, arduino_library_dir, update=False):

    # Create directories
    library_dir = os.path.join(arduino_library_dir, 'emlearn')

    if os.path.exists(library_dir) and not update:
        raise Error(f"""Library already exists at '{library_dir}'.
            Specifying --update will overwrite/update the files.""")

    ensure_dir_single(arduino_library_dir)
    ensure_dir_single(library_dir)

    def in_library(pattern):
        return os.path.join(emlearn_include_dir, pattern)

    patterns = [
        in_library('eml_*.h'), # must have
        in_library('VERSION.txt'), # convenient for debugging/support etc
    ]

    # Copy the files
    for p in patterns:
        for f in glob.glob(p):
            shutil.copy(f, library_dir+'/')

    return library_dir

def find_arduino_library_dir():
    """

    https://support.arduino.cc/hc/en-us/articles/4412950938514-Open-the-Sketchbook
    """

    platform = sys.platform
    default_platform = '*NIX'

    sketchbook_paths = {
        'win32': '~\Documents\Arduino',
        'darwin': f'~/Documents/Arduino',
        'linux': '~/Arduino',
        '*NIX': '~/Arduino',
    }

    if platform not in sketchbook_paths:
        print("WARNING: Unknown platform '{platform}'. Guessing {default_platform}")
        platform = default_platform

    sketchbook = os.path.expanduser(sketchbook_paths.get(platform))
    has_sketchbook = os.path.exists(sketchbook)
    if not has_sketchbook:
        raise Error(f"Unable to find Arduino sketchbook at '{sketchbook}'")

    libraries = os.path.join(sketchbook, 'libraries')
    return libraries
   

def parse():

    epilog = """
    In case of errors, check that you have the Arduino IDE already installed.

    If you are using a non-standard location for your Arduino libraries, specify --arduino-library-dir
    """

    parser = argparse.ArgumentParser(
                        prog='python -m emlearn.arduino.install',
                        description='Install emlearn as Arduino library',
                        epilog=epilog)

    parser.add_argument('--arduino-library-dir', default=None, metavar='DIR',
        help='Location of Arduino libraries. Defaults to the Arduino IDE default. For example.',
    )

    parser.add_argument('--emlearn-include-dir', default=get_include_dir(), metavar='DIR',
        help='Location of emlearn C code. Defaults to emlearn.include_dir from the Python module',
    )

    parser.add_argument('--update', default=False, action='store_true',
        help='Update/overwrite existing files',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse()

    emlearn_include_dir = args.emlearn_include_dir
    arduino_library_dir = args.arduino_library_dir
    if arduino_library_dir is None:
        arduino_library_dir = find_arduino_library_dir()

    print(f"Using emlearn include directory {emlearn_include_dir}")
    print(f"Using Arduino library directory {arduino_library_dir}")

    installed_dir = install_arduino_library(emlearn_include_dir, arduino_library_dir, update=args.update)
    print(f"Installed emlearn to {installed_dir}")

if __name__ == '__main__':
    main()
