#ifndef EMTREES_TEST_H
#define EMTREES_TEST_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emtrees.h>

typedef void (*EmtreesCsvCallback)(const EmtreesValue *values, int length, int row);

void
emtrees_test_read_csv(FILE *fp, EmtreesCsvCallback row_callback) {
    char buffer[1024];
    EmtreesValue values[256];
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

#endif // EMTREES_TEST_H


