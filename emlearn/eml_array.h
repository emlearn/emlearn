
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
#include <string.h> // memcpy

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
    if (!(col >= 0)) { return NULL; }
    if (!(row >= 0)) { return NULL; }

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
eml_array_shift_rows(EmlArray *self, int shift)
{
    const int rows = self->dims[1];
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);
    // trying to shift more than the size of buffer is likely an error
    EML_PRECONDITION(abs(shift) <= rows, EmlSizeMismatch);
    EML_PRECONDITION(shift <= 0, EmlUnsupported); // TODO: support also forward shifts

    const size_t row_size = self->value_size * self->dims[0];

#if 0
    EML_LOG_BEGIN("eml_array_shift_rows_begin");
    EML_LOG_ADD_INTEGER("shift", shift);
    EML_LOG_ADD_INTEGER("rows", rows);
    EML_LOG_ADD_INTEGER("row_size", row_size);
    EML_LOG_END();
#endif

    for (int i=0; i<rows+shift; i++) {
        const int source_row = i-shift;
        // Assert never below zero
        void *target = eml_array_data_2d(self, 0, i);
        void *source = eml_array_data_2d(self, 0, source_row);
        if ((target == NULL) || (source == NULL)) {
            return EmlPostconditionFailed; // XXX: should be invariant failed / generic assert
        }
#if 0
        EML_LOG_BEGIN("eml_array_shift_rows_row");
        EML_LOG_ADD_INTEGER("target_row", i);
        EML_LOG_ADD_INTEGER("source_row", source_row);
        EML_LOG_END();
#endif
        memcpy(target, source, row_size);
    }

    return EmlOk;
}

EmlError
eml_array_copy_rows(EmlArray *self, int start, const EmlArray *other)
{
    // support only 2d
    EML_PRECONDITION(self->n_dims == 2, EmlUnsupported);
    EML_PRECONDITION(other->n_dims == 2, EmlUnsupported);
    // start cannot be negative
    EML_PRECONDITION(start >= 0, EmlUnsupported);
    // must have same number of columns
    EML_PRECONDITION(other->dims[0] == self->dims[0], EmlSizeMismatch);
    // other must not be too big

//    fprintf(stderr, "start %d - %d- other_rows %d, self_rows %d \n",
//        start, start+other->dims[1], other->dims[1], self->dims[1]);
    EML_PRECONDITION((start + other->dims[1]) <= self->dims[1], EmlSizeMismatch);


    const size_t row_size = self->value_size * self->dims[0];

    for (int i=0; i<other->dims[1]; i++) {

        const int target_row = i+start;
        void *target = eml_array_data_2d(self, 0, target_row);
        void *source = eml_array_data_2d((EmlArray *)other, 0, i);
        if ((target == NULL) || (source == NULL)) {
            return EmlPostconditionFailed; // XXX: should be invariant failed / generic assert
        }
#if 0
        EML_LOG_BEGIN("eml_array_shift_rows_row");
        EML_LOG_ADD_INTEGER("target_row", i);
        EML_LOG_ADD_INTEGER("source_row", source_row);
        EML_LOG_END();
#endif
        memcpy(target, source, row_size);
    }

    return EmlOk;
}



#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_AUDIO_H
