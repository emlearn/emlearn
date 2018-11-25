
#ifndef EML_COMMON_H
#define EML_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

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

// Return if expr gives error 
#define EML_CHECK_ERROR(expr) \
    do { \
        const EmlError _e = (expr); \
        if (_e != EmlOk) { \
            return _e; \
        } \
    } while (0);


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
        return NULL;
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif // EML_COMMON_H
