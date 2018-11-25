
#include <eml_audio.h>
#include <eml_benchmark.h>

#include <stdio.h>

EmlError
bench_melspec()
{
    const int n_reps = 100;
    const EmlAudioMel mel = { 64, 0, 20000, 1024, 44100 };
    float times[n_reps];

    float input_data[EML_AUDIOFFT_LENGTH] = {0};
    float temp_data[EML_AUDIOFFT_LENGTH] = {0};

    eml_benchmark_fill(times, n_reps);

    eml_benchmark_melspectrogram(mel, input_data, temp_data, n_reps, times);
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
