
"""
Tools for getting the usage of program space (FLASH) and memory (SRAM).
"""

import os
import re
import tempfile
import subprocess
from typing import Dict
from pathlib import Path

from emlearn import common

def parse_binutils_size_c_output(stdout : str) -> Dict[str, int]:
    """
    Parse the output of GNU binutils program size, with the option -C

    Outputs are in bytes
    """

    number_field_regex = r"(.*):\s+(\d+)\sbytes" # extract field name and value
    matches = re.finditer(number_field_regex, stdout, re.MULTILINE)
    out = {}
    for m in matches:
        field, value = m.groups()
        out[field.lower()] = int(value)

    assert 'program' in out.keys(), out.keys()
    assert 'data' in out.keys(), out.keys()
    return out


def run_binutils_size(elf_file : Path, binary : str) -> Dict[str, int]:
    """
    Get the size of the "program" and "data" sections,
    using the "size" command-line tool from GNU binutils
    """

    # ensure there are no translations etc
    env = dict(os.environ) 
    env['LC_ALL'] = 'C' 

    args = [ binary, '-C', elf_file ]
    out = subprocess.check_output(args, env=env)

    parsed = parse_binutils_size_c_output(out.decode('utf-8'))

    return parsed

def build_avr8_code(code, work_dir : Path,
        mcu : str = 'atmega2560',
        makefile : Path = None,
        extra_cflags : str = '',
        make : str ='make'):

    # FIXME: improve path
    platforms_dir = os.path.join(os.path.dirname(__file__), '../platforms') 
    if makefile is None:
        makefile = os.path.join(platforms_dir, 'avr/Makefile')
    
    assert os.path.exists(makefile), f"Makefile not found: {makefile}"

    code_basename = 'prog'
    code_path = os.path.join(work_dir, f'{code_basename}.c')
    with open(code_path, 'w') as c:
        c.write(code)

    args = [
        make,
        '-f', makefile,
        f'OBJS={code_basename}.o',
        f'MCU={mcu}',
        f'BIN=out',
        f'EXTRA_CFLAGS={extra_cflags}',
    ]
    output = subprocess.check_output(args, cwd=work_dir)

    elf_path = os.path.join(work_dir, 'out.elf')
    assert os.path.exists(elf_path), os.listdir(work_dir)

    return elf_path

def get_program_size(code : str, platform : str, include_dirs=None):

    if platform != 'avr':
        # FIXME: also support "host"
        # TODO: support also ARM Cortex M0+M4F
        # TODO: support also ESP32 Xtensa
        raise NotImplementedError("Only the 'avr' platform is implemented at this time")

    size_bin = 'avr-size'

    emlearn_include_dir = common.get_include_dir()
    if include_dirs is None:
        # default
        include_dirs = [ emlearn_include_dir ]

    cflags = ' '.join([ f"-I{d}" for d in include_dirs ])

    with tempfile.TemporaryDirectory() as temp_dir:

        # build program
        try:
            elf_path = build_avr8_code(code, work_dir=temp_dir, extra_cflags=cflags)
        except subprocess.CalledProcessError as e:
            print('STDOUT', e.stdout)
            raise e

        sizes = run_binutils_size(elf_path, binary=size_bin)

    return sizes

