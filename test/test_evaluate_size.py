
import pytest
import shutil
from emlearn.evaluate.size import get_program_size, check_build_tools

PLATFORM_CPUS = [
    'avr/atmega328',
    'avr/atmega2560',
    'arm/Cortex-M0',
    'arm/Cortex-M0+',
    'arm/Cortex-M3',
    'arm/Cortex-M4F',
]

@pytest.mark.parametrize('platform_mcu', PLATFORM_CPUS)
def test_model_size(platform_mcu):

    platform, mcu = platform_mcu.split('/')
    missing_buildtools = check_build_tools(platform)

    if missing_buildtools:
        pytest.skip(str(missing_buildtools))

    # Just a simple portable program that does some computations, including floatin point
    example_program = \
    """
    #include <stdint.h>
    #include <stdbool.h>

    int function1(int a) {
        const float f = 2.0f + (a*3.3f / 7.4f);
        const int out = f < 1.5;
        return out;
    }

    int main()
    {
        volatile bool input;
        input = true;
        const int out = function1(input);
        return out;
    }
    """
    code = example_program
    sizes = get_program_size(code, platform=platform, mcu=mcu)
    assert sizes.get('flash') >= 100, sizes

