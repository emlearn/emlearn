
"""
Run the C code tests using pytest
"""

import os
import re
import shutil
import subprocess

import pytest

from emlearn.common import compile_executable

here = os.path.dirname(__file__)


C_TEST_MODULES = [
    #'signal_windower',
    'array',
    'neighbors',
    'quantizer',
    'net',
]

def parse_test_summary(stdout):
    """
    Parse output of Unity test runner
    """

    lines = stdout.strip().split('\n')

    summary_line = lines[-2]

    # get number of tests/fails
    count_regex = r'(\d+) \w* (\d+) \w* (\d+) \w*'
    match = re.match(count_regex, summary_line)
    if match is None:
        raise ValueError(f'Not a test summary line: {summary_line}')

    n_tests, n_fails, n_ignored = match.groups()
    out = {
        'tests': int(n_tests),
        'failures': int(n_fails),
        'ignored': int(n_ignored),
    }
    return out


def run_test(bin_path, module):

    env = { 'EMLEARN_TEST_MODULES': module }
    args = [
        bin_path,
    ]
    stdout = subprocess.check_output(args, env=env).decode('utf-8')

    summary = parse_test_summary(stdout)

    assert summary['tests'] > 0
    assert summary['failures'] == 0
    assert '\nOK\n' in stdout, out


@pytest.fixture()
def c_tests_executable():

    # Compile the tests code
    eml_dir = os.path.join(here, '..', 'emlearn')
    unity_dir = os.path.join(here, 'Unity', 'src')
    test_file_path = os.path.join(here, 'test_all.c')
    assert os.path.exists(test_file_path)
    out_dir = os.path.join(here, 'out', 'test_c')
    name = 'run_tests'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    include_dirs = [
        eml_dir,
        unity_dir,
    ]
    bin_path = compile_executable(test_file_path,
        out_dir=out_dir, name=name, include_dirs=include_dirs)

    yield bin_path

    # cleanup
    os.unlink(bin_path)
    os.rmdir(out_dir)


@pytest.mark.parametrize('module', C_TEST_MODULES)
def test_c_module(module, c_tests_executable):

    bin_path = c_tests_executable
    run_test(bin_path, module)
