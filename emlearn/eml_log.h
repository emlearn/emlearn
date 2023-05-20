
#ifndef EML_LOG_H
#define EML_LOG_H

#include <stddef.h>
#include <stdio.h>

// Configuration

#ifndef EML_LOG_ENABLE
#define EML_LOG_ENABLE 0
#endif

#if EML_LOG_ENABLE
// enabled

#ifndef EML_LOG_PRINTF
#define EML_LOG_PRINTF(...) fprintf(stderr, __VA_ARGS__)
#endif

#else // EML_LOG_ENABLE

// disabled, no implementation
#define EML_LOG_PRINTF(...)

#endif // EML_LOG_ENABLE

// Utilities

#define EML_LOG_PRINT_ARRAY(arr, len, fmt) \
do { \
    EML_LOG_PRINTF("["); \
    for (int i=0; i<len; i++) { \
        EML_LOG_PRINTF(fmt, arr[i]); \
        if (i != len-1) { \
            EML_LOG_PRINTF(","); \
        } \
    } \
    EML_LOG_PRINTF("]"); \
} while (0)

#define EML_LOG_BOOL_STR(b) b ? "true" : "false"  

// Structured logging API

#define EML_LOG_BEGIN(eventname)                    EML_LOG_PRINTF(eventname)
#define EML_LOG_END()                               EML_LOG_PRINTF("\n")

#define EML_LOG_ADD(key, str)                       EML_LOG_PRINTF(" %s=%s", key, str)
#define EML_LOG_ADD_INTEGER(key, integer)           EML_LOG_PRINTF(" %s=%d", key, integer)
#define EML_LOG_ADD_BOOL(key, boolean)              EML_LOG_PRINTF(" %s=%s", key, (boolean) ? "true" : "false")
#define EML_LOG_ADD_FLOAT(key, flt)                 EML_LOG_PRINTF(" %s=%.4f", key, flt)

#define EML_LOG_ADD_ARRAY(key, arr, len, fmt) \
do { \
    EML_LOG_PRINTF(" %s", key); \
    EML_LOG_PRINT_ARRAY(arr, len, fmt); \
} while (0)




#endif // EML_LOG_H
