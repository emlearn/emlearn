
#include <eml_audio.h>
#include <eml_benchmark.h>

#include <stdio.h>

EmlError
bench_melspec()
{
    const int n_reps = 100;
    const EmlAudioMel mel = { 64, 0, 20000, 1024, 44100 };
    float times[n_reps];

    eml_benchmark_fill(times, n_reps);

    eml_benchmark_melspectrogram(mel, n_reps, times);
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
