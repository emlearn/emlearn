
#include <eml_audio.h>
#include <eml_benchmark.h>

#include <stdio.h>

#ifndef EML_N_FFT
#define EML_N_FFT 1024
#define EML_N_REPS 1000
#define EML_FRAME_LENGTH 1024
#define EML_N_FFT_TABLE (EML_N_FFT/2)
#endif

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

EmlError
bench_melspec()
{
    const EmlAudioMel mel = { 64, 0, 20000, EML_N_FFT, 44100 };
    float times[EML_N_REPS];

    float input_data[EML_FRAME_LENGTH];
    float temp_data[EML_FRAME_LENGTH];
    eml_benchmark_fill(input_data, EML_FRAME_LENGTH);

    float fft_sin[EML_N_FFT_TABLE];
    float fft_cos[EML_N_FFT_TABLE];
    EmlFFT fft = { EML_N_FFT_TABLE, fft_sin, fft_cos };
    EML_CHECK_ERROR(eml_fft_fill(fft, EML_N_FFT));

    eml_benchmark_melspectrogram(mel, fft, input_data, temp_data, EML_N_REPS, times);

    const float mean = eml_signal_mean(times, EML_N_REPS);
    printf("melspec;%d;%f\n", EML_N_REPS, mean);
    return EmlOk;
}

EmlError
bench_all()
{
    printf("task;repetitions;avg_time_us\n");
    EML_CHECK_ERROR(bench_melspec());
    return EmlOk;
}

int main() {

    const EmlError e = bench_all();
    return -(int)e;
}
