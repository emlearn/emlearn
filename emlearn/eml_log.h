
#ifndef EML_LOG_H
#define EML_LOG_H

/** @file eml_log.h
* emlearn has support for structured logging.
* This is primarily used internally, but may be used by applications.
*
* A structured log entry consist of an event identifier, along with key/value pairs.
* An entry is serialized with [logfmt](https://brandur.org/logfmt):
*
* ```
* event-name key1=value1 key2=value2\n
* ```
*
* Structured logging has the following advantages:
* - Machine-readable. Easy to analyze with scripts
* - Event names should be unique and stable. Easy to search for
* - Indexable, including on particular keys/values. Can be merged with other data sources
*
* Typical usage:
* ```
* EML_LOG_BEGIN("my-event");
* EML_LOG_ADD("key1", "string-value");
* EML_LOG_ADD_INTEGER("key2", 123);
* EML_LOG_ADD_BOOL("key3", true);
* EML_LOG_ADD_FLOAT("key2", 3.14);
* EML_LOG_END();
* ```
*/

#include <stddef.h>
#include <stdio.h>

// Configuration

/**
\brief Enable logging

To enable #define EML_LOG_ENABLE 1
Default is OFF.
*/
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
/*
* \internal
Helper for printing an array
*/
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

/**
Start a log entry
*/
#define EML_LOG_BEGIN(eventname)                    EML_LOG_PRINTF(eventname)
/**
End a log entry
*/
#define EML_LOG_END()                               EML_LOG_PRINTF("\n")

/**
Add a key/value pair (strings)
*/
#define EML_LOG_ADD(key, str)                       EML_LOG_PRINTF(" %s=%s", key, str)
/**
Add an integer value
*/
#define EML_LOG_ADD_INTEGER(key, integer)           EML_LOG_PRINTF(" %s=%d", key, integer)
/**
Add a boolean value
*/
#define EML_LOG_ADD_BOOL(key, boolean)              EML_LOG_PRINTF(" %s=%s", key, (boolean) ? "true" : "false")
/**
Add a floating-point value
*/
#define EML_LOG_ADD_FLOAT(key, flt)                 EML_LOG_PRINTF(" %s=%.4f", key, flt)
/**
Add an array
*/
#define EML_LOG_ADD_ARRAY(key, arr, len, fmt) \
do { \
    EML_LOG_PRINTF(" %s", key); \
    EML_LOG_PRINT_ARRAY(arr, len, fmt); \
} while (0)



#endif // EML_LOG_H
