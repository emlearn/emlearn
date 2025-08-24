 
/*
eml_fileio.h

Utilities for I/O streams using FILE objects from C standard library.
Useful together with EmlCsvReader / EmlCsvWriter and related.
*/

#ifndef EML_FILEIO_H
#define EML_FILEIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

/**
* \brief Implements EmlIoWriteFunction for FILE stream
*/
int eml_fileio_write(void *context, const uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fwrite(buffer, 1, size, fptr);
}

/**
* \brief Implements EmlIoReadFunction for FILE stream
*/
int eml_fileio_read(void *context, uint8_t *buffer, size_t size)
{
    FILE* fptr = context;
    return fread(buffer, 1, size, fptr);
}

/**
* \brief Implements EmlIoSeekFunction for FILE stream
*/
int eml_fileio_seek(void *context, size_t position)
{
    FILE* fptr = context;
    return fseek(fptr, position, SEEK_SET);
}

#ifdef __cplusplus
}
#endif

#endif // EML_FILEIO_H
