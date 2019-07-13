
import numpy
from emlearn import data

def main():

    a = numpy.int32([[11, 222], [3333, 4444444]])
    d = data.serialize(a)

    print(d)
    with open('out.emld', 'wb+') as f:
        f.write(d)


if __name__ == '__main__':
    main()
