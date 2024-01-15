
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


platforms_dir = os.path.join(os.path.dirname(__file__), '../platforms') 

# Compiler options for particular ARM Cortex families of CPUs
# Ref https://github.com/ARM-software/toolchain-gnu-bare-metal/blob/master/readme.txt
ARM_CORTEX_CFLAGS = {
    'Cortex-M0': '-mthumb -mcpu=cortex-m0',
    'Cortex-M0+': '-mthumb -mcpu=cortex-m0plus',
    'Cortex-M3': '-mthumb -mcpu=cortex-m3',
    'Cortex-M4F': '-mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard',
}


def parse_binutils_size_a_output(stdout : str) -> Dict[str, int]:
    """
    Parse the output of GNU binutils program size, with the option -A

    Outputs are in bytes
    """
    # NOTE: the -C format is a bit more directly what we want,
    # but it is not supported on for example arm-none-eabi-size

    # Do some sanity checks on the output
    lines = stdout.split('\n')
    file_line = lines[0]
    header_line = lines[1]
    assert '.elf' in file_line, file_line
    assert 'section' in header_line, header_line
    assert 'size' in header_line, header_line

    # extract section names and sizes
    number_field_regex = r"([\w.]+)\s+(\d+)" 
    matches = re.finditer(number_field_regex, stdout, re.MULTILINE)
    out = {}
    for m in matches:
        field, value = m.groups()
        out[field.lower()] = int(value)

    # It can happen that there is no RAM/stack usage
    if '.data' not in out:
        out['.data'] = 0

    assert '.text' in out.keys(), out.keys()
    assert '.data' in out.keys(), out.keys()
    return out

def run_binutils_size(elf_file : Path, binary : str) -> Dict[str, int]:
    """
    Get the size of the "program" and "data" sections,
    using the "size" command-line tool from GNU binutils
    """

    # ensure there are no translations etc
    env = dict(os.environ) 
    env['LC_ALL'] = 'C' 

    args = [ binary, '-A', elf_file ]
    out = subprocess.check_output(args, env=env)

    parsed = parse_binutils_size_a_output(out.decode('utf-8'))

    converted = {
        'program': parsed['.text'],
        'ram': parsed['.data'],
    }

    return converted

def build_avr8_code(code, work_dir : Path,
        mcu : str = 'atmega2560',
        makefile : Path = None,
        extra_cflags : str = '',
        make : str ='make'):

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


def build_arm_cortex_code(code, work_dir : Path,
        mcu : str = 'Cortex-M0',
        makefile : Path = None,
        extra_cflags : str = '',
        make : str ='make'):

    if makefile is None:
        makefile = os.path.join(platforms_dir, 'arm-cortex/Makefile')
    
    assert os.path.exists(makefile), f"Makefile not found: {makefile}"

    code_basename = 'prog'
    code_path = os.path.join(work_dir, f'{code_basename}.c')
    with open(code_path, 'w') as c:
        c.write(code)

    supported_mcus = set(ARM_CORTEX_CFLAGS.keys())
    if mcu not in supported_mcus:
        raise ValueError(f"Unsupported ARM CPU type: '{mcu}'. Options: {','.join(supported_mcus)}")
    arm_cflags = ARM_CORTEX_CFLAGS[mcu]

    args = [
        make,
        '-f', makefile,
        f'OBJS={code_basename}.o',
        f'BIN=out',
        f'ARM_CORTEX_CFLAGS={arm_cflags}',
        f'EXTRA_CFLAGS={extra_cflags}',
    ]
    output = subprocess.check_output(args, cwd=work_dir)

    elf_path = os.path.join(work_dir, 'out.elf')
    assert os.path.exists(elf_path), os.listdir(work_dir)

    return elf_path

PLATFORM_BUILDERS = {
    'avr': build_avr8_code,
    'arm-cortex': build_arm_cortex_code,
}
PLATFORM_SIZE_COMMAND = {
    'avr': 'avr-size',
    'arm-cortex': 'arm-none-eabi-size',
}

def get_program_size(code : str, platform : str, include_dirs=None):

    supported_platforms = set(PLATFORM_BUILDERS.keys())
    if platform not in supported_platforms:
        # FIXME: also support "host"
        # TODO: support also ARM Cortex M0+M4F
        # TODO: support also ESP32 Xtensa
        raise ValueError("Unsupported build platform '{platform}'. Supported: {','.join(supported_platforms)}")

    build_function = PLATFORM_BUILDERS[platform]
    size_bin = PLATFORM_SIZE_COMMAND[platform]

    emlearn_include_dir = common.get_include_dir()
    if include_dirs is None:
        # default
        include_dirs = [ emlearn_include_dir ]

    cflags = ' '.join([ f"-I{d}" for d in include_dirs ])

    with tempfile.TemporaryDirectory() as temp_dir:

        # build program
        try:
            elf_path = build_function(code, work_dir=temp_dir, extra_cflags=cflags)
        except subprocess.CalledProcessError as e:
            print('STDOUT', e.stdout)
            raise e

        sizes = run_binutils_size(elf_path, binary=size_bin)

    return sizes

