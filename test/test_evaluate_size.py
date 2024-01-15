
import pytest
import shutil
from emlearn.evaluate.size import get_program_size

def check_programs(programs):
    needed = set(programs)
    have = set([ p for p in needed if shutil.which(p) ])
    missing = needed - have
    if missing == set():
        return None

    return f"Missing programs: {', '.join(missing)}"

missing_avr_buildtools = check_programs(['avr-size', 'avr-gcc', 'make',])

# TODO: test all supported platforms. Different ARM Cortex M etc
# FIXME: move the dependency checks into size.py
@pytest.mark.skipif(bool(missing_avr_buildtools), reason=str(missing_avr_buildtools))
def test_model_size_avr8():

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
        const int out = function1(input);
        return out;
    }
    """
    code = example_program
    sizes = get_program_size(code, platform='avr')
    assert sizes.get('program') >= 100, sizes

