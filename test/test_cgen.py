
import pytest

from emlearn.cgen import identifier_is_reserved, identifier_is_valid, assert_valid_identifier

IDENTIFIERS = dict([
    # identifier, valid
    ('plain_with_underscore', True),
    ('UpperCase_is_OK', True),
    ('_number3', True),
    ('cannot-use-minus', False),
    ('equal=invalid', False),
    ('plus+is+not+positive', False),
    ('cannot have spaces', False),
    ('else', False), # C89
    ('inline', False), # C99
    ('_Noreturn', False),
    ('true', False), # C23
])

@pytest.mark.parametrize('name', IDENTIFIERS.keys())
def test_cgen_check_identifiers(name):
    identifier = name
    expect_valid = IDENTIFIERS[name]

    is_not_reserved = not identifier_is_reserved(identifier)
    is_valid = identifier_is_valid(identifier)

    if expect_valid:
        assert is_valid == True
        assert is_not_reserved == True

    else:
        assert (not is_valid) or not (is_not_reserved), (is_valid, is_not_reserved)

    if expect_valid:
        assert_valid_identifier(identifier)
    else:
        with pytest.raises(Exception) as e_info:
            assert_valid_identifier(identifier)

