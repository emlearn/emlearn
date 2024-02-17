
"""
Test that library works when used with Arduino IDE
"""

import os
import shutil
import re
import subprocess

from emlearn.arduino.install import install_arduino_library
from emlearn.common import get_include_dir
from emlearn.utils.fileutils import ensure_dir

import pytest

here = os.path.dirname(__file__)

arduino_cli_bin = 'arduino-cli'
have_arduino_cli = shutil.which(arduino_cli_bin) is not None
# when this envvar is set, then we must run the test
force_arduino_test = bool(int(os.environ.get('EMLEARN_TEST_ARDUINO', '0')))
enable_arduino_tests = have_arduino_cli or force_arduino_test
skip_reason = f'{arduino_cli_bin} not found'

def remove_ansi_escapes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    out = ansi_escape.sub('', text)
    return out


def arduino_build(sketch_dir, library_dir, board='arduino:avr:uno'):

    args = [
        'arduino-cli',
        'compile',
        '-b', board,
        sketch_dir,
        '--library', library_dir, # prepend library custom path
    ]
    try:
        out = subprocess.check_output(args)
    except subprocess.CalledProcessError as e:
        print(remove_ansi_escapes(e.stdout.decode('utf-8')))
        raise e

    cleaned = remove_ansi_escapes(out.decode('utf-8'))
    return cleaned

@pytest.mark.skipif(not enable_arduino_tests, reason=skip_reason)
def test_arduino_preconditions():
    assert have_arduino_cli, ''


@pytest.mark.skipif(not enable_arduino_tests, reason=skip_reason)
def test_arduino_helloworld():
    out_dir = os.path.join(here, 'out/arduino_helloworld/helloworld_xor')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    example_dir = os.path.join(here, '../docs/helloworld_xor/')
    sketch_path = os.path.join(example_dir, 'helloworld_xor.ino')
    train_path = os.path.abspath(os.path.join(example_dir, 'xor_train.py'))
    emlearn_dir = get_include_dir()
    ensure_dir(out_dir)
    library_dir = os.path.join(out_dir, 'libs')

    # install library
    # using a custom path to avoid messing with peoples files
    install_arduino_library(emlearn_dir, arduino_library_dir=library_dir)

    # build the model
    subprocess.check_output(['python', train_path], cwd=out_dir)

    # copy sketch into clean directory
    shutil.copy(sketch_path, out_dir)

    # build sketch
    out = arduino_build(out_dir, library_dir=os.path.join(library_dir, 'emlearn'))
    assert 'Sketch uses' in out
    assert 'Used library'
    assert 'emlearn' in out
    assert library_dir in out


