/* 
 * Free FFT and convolution (C)
 * 
 * Copyright (c) 2017 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 *
 * Copyright (c) 2018 Jon Nordby. (MIT License)
 * Modifications to run without malloc, on single-precision float
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

#ifndef EML_FFT_H
#define EML_FFT_H

#ifdef __cplusplus
extern "C" {
#endif


#include <math.h>

#include "eml_common.h"

static size_t reverse_bits(size_t x, int n) {
	size_t result = 0;
	for (int i = 0; i < n; i++, x >>= 1)
		result = (result << 1) | (x & 1U);
	return result;
}

typedef struct _EmlFFT {
    int length; // (n/2)
    float *sin;
    float *cos;
} EmlFFT;


EmlError
eml_fft_fill(EmlFFT table, size_t n) {
    EML_PRECONDITION((size_t)table.length == n/2, EmlSizeMismatch);

	// Trignometric tables
	for (size_t i = 0; i < (size_t)(n / 2); i++) {
		table.cos[i] = (float)cos(2 * M_PI * i / n);
		table.sin[i] = (float)sin(2 * M_PI * i / n);
	}
    return EmlOk;
}

EmlError
eml_fft_forward(EmlFFT table, float real[], float imag[], size_t n) {

    // Compute levels = floor(log2(n))
	int levels = 0;
	for (size_t temp = n; temp > 1U; temp >>= 1)
		levels++;

    EML_PRECONDITION(((size_t)(1U << levels)) == n, EmlSizeMismatch);
    EML_PRECONDITION((size_t)table.length == n/2, EmlSizeMismatch);

	// Bit-reversed addressing permutation
	for (size_t i = 0; i < n; i++) {
		size_t j = reverse_bits(i, levels);
		if (j > i) {
			float temp = real[i];
			real[i] = real[j];
			real[j] = temp;
			temp = imag[i];
			imag[i] = imag[j];
			imag[j] = temp;
		}
	}
	
	// Cooley-Tukey decimation-in-time radix-2 FFT
	for (size_t size = 2; size <= n; size *= 2) {
		size_t halfsize = size / 2;
		size_t tablestep = n / size;
		for (size_t i = 0; i < n; i += size) {
			for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				size_t l = j + halfsize;
				float tpre =  real[l] * table.cos[k] + imag[l] * table.sin[k];
				float tpim = -real[l] * table.sin[k] + imag[l] * table.cos[k];
				real[l] = real[j] - tpre;
				imag[l] = imag[j] - tpim;
				real[j] += tpre;
				imag[j] += tpim;
			}
		}
		if (size == n)  // Prevent overflow in 'size *= 2'
			break;
	}	
	return EmlOk;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_FFT_H
