
#include <eml_audio.h>
#include <eml_benchmark.h>

#include <stdio.h>

int main() {

    const int n_reps = 100;
    const EmlAudioMel mel = { 64, 0, 20000, 1024, 44100 };
    float times[n_reps];

    eml_benchmark_fill(times, n_reps);

    eml_benchmark_melspectrogram(mel, n_reps, times);
    EmlVector t = { times, n_reps };

    const float mean = eml_vector_mean(t);
    printf("melspec;%d;%f\n", n_reps, mean);
}
