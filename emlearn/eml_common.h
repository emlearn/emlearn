
#ifndef EML_COMMON_H
#define EML_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// EmlError
typedef enum _EmlError {
    EmlOk = 0,
    EmlSizeMismatch,
    EmlUnsupported,
    EmlUninitialized,
    EmlPostconditionFailed,
    EmlUnknownError,
    EmlErrors,
} EmlError;

static const char *
eml_error_strs[EmlErrors] = {
    "OK",
    "SizeMismatch",
    "Uninitialized",
    "Unsupported",
    "Postcondition failed",
    "Unknown error",
};

bool
eml_error_valid(EmlError e) {
    return (e >= EmlOk && e < EmlErrors);
}

const char *
eml_error_str(EmlError e) {
    if (eml_error_valid(e)) {
        return eml_error_strs[e];
    } else {
        return "Invalid error code";
    }
}


typedef void (*EmlDebugFunction)(EmlError code, const char *file, int line);

#ifdef EML_DEBUG
void eml_debug_stderr(EmlError code, const char *file, int line)
{
    const bool log_error = getenv("EML_DEBUG_LOG") != NULL;
    const bool fatal_errors = getenv("EML_DEBUG_FATAL_ERRORS") != NULL;
    if (log_error) {
        const char *err_str = eml_error_str(code);
        fprintf(stderr, "EmlError %s at %s:%d \n",
                        err_str, file, line);
    }
    if (fatal_errors) {
        exit(11);
    }
}

static EmlDebugFunction eml_debug = eml_debug_stderr;
#endif

#ifdef EML_DEBUG

#define EML_PRECONDITION(expr, errorcode) \
    do { \
        if (!(expr)) { \
            if (eml_debug) { eml_debug(errorcode, __FILE__, __LINE__); } \
            return errorcode; \
        } \
    } while (0);

// Return errorcode if expr not satisfied
#define EML_POSTCONDITION(expr, errorcode) \
    do { \
        if (!(expr)) { \
            if (eml_debug) { eml_debug(errorcode, __FILE__, __LINE__); } \
            return errorcode; \
        } \
    } while (0);


#else // EML_DEBUG


// Return errorcode if expr not satisfied
#define EML_PRECONDITION(expr, errorcode) \
    do { \
        if (!(expr)) { \
            return errorcode; \
        } \
    } while (0);

// Return errorcode if expr not satisfied
#define EML_POSTCONDITION(expr, errorcode) \
    do { \
        if (!(expr)) { \
            return errorcode; \
        } \
    } while (0);

#endif // EML_DEBUG

// Return if expr gives error
#define EML_CHECK_ERROR(expr) \
    do { \
        const EmlError _e = (expr); \
        if (_e != EmlOk) { \
            return _e; \
        } \
    } while (0);



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static inline
float eml_max(float a, float b) {
    return (a > b) ? a : b;
}
static inline
float eml_min(float a, float b) {
    return (a < b) ? a : b;
}

#ifdef __cplusplus
}
#endif


#endif // EML_COMMON_H
