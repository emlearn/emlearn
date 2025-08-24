
#ifndef EML_IO_H
#define EML_IO_H

#ifdef __cplusplus
extern "C" {
#endif

// I/O functions for streams
typedef int (*EmlIoReadFunction)(void *context, uint8_t *buffer, size_t size);
typedef int (*EmlIoSeekFunction)(void *context, size_t position);
typedef int (*EmlIoWriteFunction)(void *context, const uint8_t *buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif // EML_IO_H
