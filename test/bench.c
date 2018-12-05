
#include <eml_audio.h>
#include <eml_benchmark.h>

#include <stdio.h>

EmlError
bench_melspec()
{
    const int n_fft = 1024;
    const int n_reps = 1000;
    const EmlAudioMel mel = { 64, 0, 20000, n_fft, 44100 };
    float times[n_reps];

    const int frame_length = 1024;

    float input_data[frame_length];
    float temp_data[frame_length];
    eml_benchmark_fill(input_data, frame_length);

    const int n_fft_table = n_fft/2;
    float fft_sin[n_fft_table];
    float fft_cos[n_fft_table];
    EmlFFT fft = { n_fft_table, fft_sin, fft_cos };
    EML_CHECK_ERROR(eml_fft_fill(fft, n_fft));

    eml_benchmark_melspectrogram(mel, fft, input_data, temp_data, n_reps, times);
    EmlVector t = { times, n_reps };

    const float mean = eml_vector_mean(t);
    printf("melspec;%d;%f\n", n_reps, mean);
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
