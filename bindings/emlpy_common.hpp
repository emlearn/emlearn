
#ifndef EMLPY_COMMON
#define EMLPY_COMMON

#include <eml_common.h>

#include <exception>

namespace emlpy {

class PreconditionFailed : public std::runtime_error {
public:
    PreconditionFailed(const char *str)
        : std::runtime_error(str)

    {
    }

};


class Error : public std::runtime_error {
public:
    Error(EmlError err)
        : std::runtime_error(eml_error_str(err))
    {
    }

};


} // end namespace emlpy



// Throw if expr is false
#define EMLPY_PRECONDITION(expr, msg) \
    do { \
        if (!(expr)) { \
            throw new emlpy::PreconditionFailed(msg); \
        } \
    } while (0);

// Throw if expr gives error
#define EMLPY_CHECK_ERROR(expr) \
    do { \
        const EmlError _e = (expr); \
        if (_e != EmlOk) { \
            throw new emlpy::Error(_e); \
        } \
    } while (0);


#endif // EMLPY_COMMON
