#ifndef EML_TEST_H
#define EML_TEST_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eml_common.h"

typedef void (*EmlCsvCallback)(const float *values, int length, int row);

// Return number of values parsed, or -EmlError
int32_t
eml_test_parse_csv_line(char *buffer, float *values, int32_t values_length,
                        int32_t *values_read_out)
{        
    EML_PRECONDITION(buffer, EmlUninitialized);
    EML_PRECONDITION(values, EmlUninitialized);

    int field_no = 0;
    const char seps[] = ",;";
    char *token = strtok(buffer, seps);

    while (token != NULL)
    {
        float value;
        sscanf(token, "%f", &value);

        if (field_no >= values_length) {
            return EmlSizeMismatch;
        }

        values[field_no] = value; 
        field_no++;
        token = strtok(NULL, seps);
    }

    if (values_read_out) {
        *values_read_out = field_no;
    }

    return EmlOk;
}

EmlError
eml_test_read_csv(FILE *fp, EmlCsvCallback row_callback) {
    const int32_t buffer_length = 1024;
    char buffer[buffer_length];
    const int32_t values_length = 256;
    float values[values_length];
    int row_no = 0;

    while(fgets(buffer, buffer_length, fp))
    {
        int value_no = 0;
        const EmlError e = eml_test_parse_csv_line(buffer, values, values_length, &value_no);
        EML_CHECK_ERROR(e);
        row_callback(values, value_no, row_no);
        row_no++;
    }
    return EmlOk;
}


#endif // EML_TEST_H


