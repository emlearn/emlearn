

def struct_init(*args):
    return '{ ' + ', '.join(str(a) for a in args) + ' }'


def constant(val, dtype='float'):
    if dtype == 'float':
        return "{:f}f".format(val)
    else:
        return str(val)


def array_declare(name, size, dtype='float', modifiers='static const',
                    values=None, end='', indent=''):
    init = ''
    if values is not None:
        init_values = ', '.join(constant(v, dtype) for v in values)
        init = ' = {{ {init_values} }}'.format(**locals())

    return '{indent}{modifiers} {dtype} {name}[{size}]{init};{end}'.format(**locals())
