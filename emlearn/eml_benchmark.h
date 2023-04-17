
#ifndef EML_BENCHMARK_H
#define EML_BENCHMARK_H
#ifdef __cplusplus
extern "C" {
#endif

#include "eml_common.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
// Unix-like system
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#define EML_HAVE_SYS_TIME 1
#endif


#ifdef EML_HAVE_SYS_TIME
#include <sys/time.h>
int64_t eml_benchmark_micros() {
    struct timeval spec;
    gettimeofday(&spec, NULL);
    //struct timespec spec; 
    //clock_gettime(CLOCK_MONOTONIC, &spec);
    const int64_t micros = (int64_t)(spec.tv_sec)*1000LL*1000LL + spec.tv_usec;
    return micros;
}
#endif

#ifdef _WIN32
#include <windows.h>
int64_t eml_benchmark_micros(void)
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    double sec = (double)t.QuadPart/(double)f.QuadPart;
    return (int64_t)(sec * 1000000LL);
}
#endif

#ifdef ARDUINO
int64_t eml_benchmark_micros() {
    return micros();
}
#endif

// https://en.wikipedia.org/wiki/Lehmer_random_number_generator#Parameters_in_common_use
static uint32_t
eml_lcg_parkmiller(uint32_t *state) {
    const uint32_t N = 0x7fffffff;
    const uint32_t G = 48271u;

    uint32_t div = *state / (N / G);
    uint32_t rem = *state % (N / G);

    uint32_t a = rem * G;
    uint32_t b = div * (N % G);

    return *state = (a > b) ? (a - b) : (a + (N - b));
}

EmlError
eml_benchmark_fill(float *values, int features) {
    uint32_t rng_state = 1;    

    for (int i=0; i<features; i++) {
        values[i] = (float)eml_lcg_parkmiller(&rng_state);
    }
    return EmlOk;
}


EmlError
eml_benchmark_melspectrogram(EmlAudioMel mel_params,
                    EmlFFT fft,
                    float *input_data, float *temp_data, 
                    int n_repetitions, float *times)
{
    // prepare data
    EmlVector input = { input_data, mel_params.n_fft };
    EmlVector temp = { temp_data, mel_params.n_fft };

    // run tests
    float sum = 0;
    for (int i=0; i<n_repetitions; i++) {
        const int64_t start = eml_benchmark_micros();

        eml_audio_melspectrogram(mel_params, fft, input, temp);
        sum += input.data[0];

        const int64_t end = eml_benchmark_micros();
        times[i] = (float)(end - start);
    }

    return EmlOk;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_BENCHMARK_H
