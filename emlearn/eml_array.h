
/*
EmlArray. An N-dimensional array container.

Similar to numpy.ndarray, but mostly meant to handle storage.
Operations should typically access the data in the array,
and use standard C types float/int16_t whenever possible.
*/

#ifndef EML_ARRAY_H
#define EML_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <eml_common.h>

#define EML_ARRAY_MAX_DIMS 3

typedef struct EmlArray_ {

    int32_t dims[EML_ARRAY_MAX_DIMS];
    int8_t n_dims;
    int8_t value_size;
    uint8_t *data;

} EmlArray;

size_t
eml_storage_size_array(int8_t n_dims, int32_t *dims, int8_t value_size)
{

    int64_t product = 1;
    for (int d=0; d<n_dims; d++) {
        product = product * dims[d]; 
    }
    return product * value_size;

}

EmlError
eml_array_init_full(EmlArray *self,
        int8_t n_dims, int32_t *dimensions,
        int8_t value_size,
        uint8_t *buffer, size_t buffer_length)
{
    EML_PRECONDITION(n_dims <= EML_ARRAY_MAX_DIMS, EmlSizeMismatch);
    EML_PRECONDITION(n_dims >= 1, EmlSizeMismatch);
    EML_PRECONDITION(dimensions[0] >= 1, EmlSizeMismatch);

    const size_t buffer_needed = eml_storage_size_array(n_dims, dimensions, value_size);
    EML_PRECONDITION(buffer_needed <= buffer_length, EmlSizeMismatch);

    /* Set dimensions */
    for (int d=0; d<EML_ARRAY_MAX_DIMS; d++) {
        self->dims[d] = 0;
    }

    self->n_dims = n_dims;
    for (int d=0; d<n_dims; d++) {
        self->dims[d] = dimensions[d];
        EML_PRECONDITION(self->dims[d] >= 0, EmlSizeMismatch);
    }

    self->value_size = value_size;

    /* Reference data */
    self->data = buffer;

    return EmlOk;
}

#define EML_ARRAY_INIT_2D(_arr, _dim1, _dim2, _values) \
do { \
     eml_array_init_full(_arr, \
        2, \
        (int32_t[EML_ARRAY_MAX_DIMS]){ _dim1, _dim2, 0 }, \
        sizeof(_values[0]), \
        (uint8_t *)(_values), \
        sizeof(_values[0])*_dim1*_dim2); \
} while (0)

size_t
eml_array_storage_size(EmlArray *self)
{
    return eml_storage_size_array(self->n_dims, self->dims, self->value_size);
}



void *
eml_array_data_2d(EmlArray *self, int col, int row)
{
/*
    EML_PRECONDITION(self->n_dims == 2, NULL);
    EML_PRECONDITION(col < self->dims[0], NULL);
    EML_PRECONDITION(row < self->dims[1], NULL);
*/
    if (!(self->n_dims == 2)) { return NULL; }
    if (!(col < self->dims[0])) { return NULL; }
    if (!(row < self->dims[1])) { return NULL; }

    const int offset = (row * self->dims[0]) + col;
    void *data = self->data + (self->value_size * offset);

    return data;
}

EmlError
eml_array_fill(EmlArray *self, float value)
{
    // FIXME: support 1d and 3d
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);
    EML_PRECONDITION(self->value_size == 2, EmlUnsupported);

    for (int i=0; i<self->dims[0]; i++) {
        for (int j=0; j<self->dims[1]; j++) {
            int16_t *val = (int16_t *)eml_array_data_2d(self, i, j);
            *val = (int16_t)value;
        }
    }
    return EmlOk;
}

EmlError
eml_array_sum(EmlArray *self, float *out_sum)
{
    // FIXME: support 1d and 3d
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);
    EML_PRECONDITION(self->value_size == 2, EmlUnsupported);

    float sum = 0.0;
    for (int i=0; i<self->dims[0]; i++) {
        for (int j=0; j<self->dims[1]; j++) {
            int16_t *data = (int16_t *)eml_array_data_2d(self, i, j);
            float val = (float)*data;
            sum += val;
        }
    }
    *out_sum = sum;
    return EmlOk;
}

EmlError
eml_array_append(EmlArray *self, EmlArray *other)
{
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);

    return EmlOk;
}

EmlError
eml_array_shift_axis(EmlArray *self, int shift)
{
    const int length = self->dims[0];
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);
    // trying to shift more than the size of buffer is likely an error
    EML_PRECONDITION(abs(shift) <= length, EmlSizeMismatch);

    if (shift < 0) {
        const int from_start = shift;
        const int to_start = 0;
        const int to_end = length - shift; 

        //memcpy();
    } else {
        const int from = 0;
        const int to = shift;
    }

    return EmlOk;
}



#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_AUDIO_H
