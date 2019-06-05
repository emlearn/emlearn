
"""Utilities to generate C code"""

FLOATING_DTYPE = 'float'
FLOATING_PRECISION = 6

def struct_init(*args):
    """Struct initializer

    >>> from emlearn import cgen
    >>> cgen.struct_init([ 1, 2, 3 ])
    "{ 1, 2, 3 }"
    """

    return '{ ' + ', '.join(str(a) for a in args) + ' }'


def constant(val, dtype='float'):
    """A literal value

    >>> from emlearn import cgen
    >>> cgen.constant(3.14)
    "3.14f"
    """
    if dtype == 'float':
        precision_formater = "{{:.{}f}}".format(FLOATING_PRECISION)
        formatted_value = precision_formater.format(val)
        formatted_value += 'f' if FLOATING_DTYPE=='float' else ''
        return formatted_value
    else:
        return str(val)


def array_declare(name, size, dtype='float', modifiers='static const',
                    values=None, end='', indent=''):

    """
    Declare and optionally initialize an array.


    >>> from emlearn import cgen
    >>> cgen.array_declare("declareonly", 10)
    "static const float declareonly[10];"

    Also intialize it.

    >>> from emlearn import cgen
    >>> cgen.array_declare("initialized", 3, dtype='int', modifiers='const')
    "const float initialized[3] = { 1, 2, 3 };"
    """
    init = ''
    if values is not None:
        init_values = ', '.join(constant(v, dtype) for v in values)
        init = ' = {{ {init_values} }}'.format(**locals())
    if dtype=='float': dtype=FLOATING_DTYPE
    return '{indent}{modifiers} {dtype} {name}[{size}]{init};{end}'.format(**locals())
