
"""
Test that library works when used with Zephyr RTOS
"""

import os
import shutil
import re
import subprocess

from emlearn.common import get_include_dir
from emlearn.utils.fileutils import ensure_dir

import pytest

here = os.path.dirname(__file__)

west_bin = 'west'
have_west = shutil.which(west_bin) is not None
# when this envvar is set, then we must run the test
force_zephyr_test = bool(int(os.environ.get('EMLEARN_TEST_ZEPHYR', '0')))
enable_zephyr_tests = have_west or force_zephyr_test
skip_reason = f'{west_bin} not found'

def west_update(workspace):

    out = subprocess.check_output(['west', 'update'], cwd=workspace)
    return out

def west_build(workspace, application, board='qemu_cortex_m0', target=None):

    args = [
        west_bin,
        'build',
        '--board', board,
        application,
    ]
    if target is not None:
        args += [ '-t', target ]

    try:
        out = subprocess.check_output(args, cwd=workspace)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode('utf-8'))
        raise e

    cleaned = out.decode('utf-8')
    return cleaned

@pytest.mark.skipif(not enable_zephyr_tests, reason=skip_reason)
def test_zephyr_preconditions():
    assert have_west, ''


@pytest.mark.skipif(not enable_zephyr_tests, reason=skip_reason)
def test_zephyr_helloworld():
    example = 'helloworld_xor'
    out_dir = os.path.join(here, 'out/zephyr_helloworld_xor')
    build_dir = os.path.join(out_dir, 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    examples_dir = os.path.join(here, '../platform_examples/zephyr/')
    shutil.copytree(examples_dir, out_dir, dirs_exist_ok=True)
    workspace_dir = out_dir

    train_path = os.path.abspath(os.path.join(here, '../docs/helloworld_xor/xor_train.py'))

    # build the model
    subprocess.check_output(['python', train_path], cwd=os.path.join(out_dir, example, 'src'))

    # Download Zephyr modules
    # might take a few minutes
    west_update(workspace_dir)

    # Build the code
    west_build(workspace_dir, example)

    # TODO: run in qemu

