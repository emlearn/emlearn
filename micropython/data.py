
import numpy
import struct

type_numbers = {
    'float32': 0,
    'int32': 1,
}

def persist(array):
    assert len(array.shape) <= 4, 'Only 1D-4D arrays are supported' 
    
    if numpy.issubdtype(array.dtype, numpy.signedinteger):
        array = array.astype(numpy.int32, casting='safe')
        type_name = 'int32'
    elif numpy.isreal(array):
        array = array.astype(float, casting='safe')
        type_name = 'float32'
    else:
        raise ValueError(f'Unsupported Numpy datatype {array.dtype}')


    # Stored as little-endian
    def uint16(v):
        return struct.pack('>H', v)
    def uint8(v):
        return struct.pack('>B', v)
    def float32(v):
        return struct.pack('f', v)
    def int32(v):
        return struct.pack('>l', v)

    # Header
    magic = b'\x93EMLEARN'
    version = uint8(1)
    dtype = uint8(type_numbers[type_name])

    dims = b''
    for i in range(0, 4):
        d = 0 if i >= len(array.shape) else array.shape[i]
        b = uint16(d)
        dims += b
    assert len(dims) == 4*2 # must also update on C side

    header = magic + version + dtype + dims + b'0'
    assert len(header) == 19 # must also update on C side

    # Send data
    flat = array.flatten(order='C')
    data = b''

    data_converter = locals()[type_name]

    for i, value in enumerate(flat):
        b = data_converter(value)
        #print('f', value, b)
        data += b

    return header + data


def main():

    #a = numpy.float32([[1.1, 2.2], [3.3, 4.4]])

    #a = numpy.int32([[11, 222], [3333, 4444444]])

    a = numpy.ndarray(shape=(2,4), dtype=numpy.int32)


    #a = numpy.ndarray(shape=(2,3,2,1), dtype=numpy.int32)

    #print(a)

    d = persist(a)

    print(d)
    with open('out.emld', 'wb+') as f:
        f.write(d)



if __name__ == '__main__':
    main()
