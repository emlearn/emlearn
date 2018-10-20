#ifndef EML_TEST_H
#define EML_TEST_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef void (*EmlCsvCallback)(const int32_t *values, int length, int row);

void
eml_test_read_csv(FILE *fp, EmlCsvCallback row_callback) {
    char buffer[1024];
    int32_t values[256];
    int row_no = 0;
    int value_no = 0;

    while(fgets(buffer, sizeof buffer, fp))
    {
        char seps[] = ",;";
        char *token = strtok(buffer, seps);
        while (token != NULL)
        {
            long value;
            sscanf(token, "%ld", &value);
            values[value_no++] = value;
            token = strtok (NULL, seps);
        }
        row_callback(values, value_no, row_no);
        value_no = 0;
        row_no++;
    }
}

// FIXME: Remove in favor of float/int32 support
typedef void (*EmBayesCsvCallback)(const val_t values[], int length, int row);
void
eml_bayes_test_read_csv(FILE *fp, EmBayesCsvCallback row_callback) {
    char buffer[1024];
    val_t values[256];
    int row_no = 0;
    int value_no = 0;

    while(fgets(buffer, sizeof buffer, fp))
    {
        char seps[] = ",;";
        char *token = strtok(buffer, seps);
        while (token != NULL)
        {
            float value;
            sscanf(token, "%f", &value);
            values[value_no++] = VAL_FROMFLOAT(value);
            token = strtok (NULL, seps);
        }
        row_callback(values, value_no, row_no);
        value_no = 0;
        row_no++;
    }
}

#endif // EML_TEST_H


