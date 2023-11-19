
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

@pytest.mark.skipif(bool(missing_avr_buildtools), reason=str(missing_avr_buildtools))
def test_model_size_avr8():

    avr_example_program = \
    """
    #include <stdbool.h>
    #include <avr/io.h>
    #include <util/delay.h>

    int main()
    {
        // set PINB0 to output in DDRB
        DDRB |= 0b00000001;

        // Set input
        DDRB &= ~(1 << PINB4);

        const bool pin_state = (PINB & (1 << PINB4)) >> PINB4;

        const float f = 2.0+ (pin_state*3.3 / 7.4);
        const int out = f < 1.5;     

        // set PINB0 low
        PORTB &= 0b11111110 + out;
        _delay_ms(500);
    }
    """
    code = avr_example_program
    sizes = get_program_size(code, platform='avr')
    assert sizes['program'] >= 1000, sizes

