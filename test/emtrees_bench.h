
#ifndef EMTREES_BENCH

// https://en.wikipedia.org/wiki/Lehmer_random_number_generator#Parameters_in_common_use
static uint32_t
lcg_parkmiller(uint32_t *state) {
    const uint32_t N = 0x7fffffff;
    const uint32_t G = 48271u;

    uint32_t div = *state / (N / G);
    uint32_t rem = *state % (N / G);

    uint32_t a = rem * G;
    uint32_t b = div * (N % G);

    return *state = (a > b) ? (a - b) : (a + (N - b));
}

void
emtrees_bench_fill(EmtreesValue *values, int rows, int features) {
    uint32_t rng_state = 1;    

    for (int row=0; row<rows; row++) {
        for (int feature=0; feature<features; feature++) {
            const int idx = (row * features) + feature;
            values[idx] = (int32_t)lcg_parkmiller(&rng_state);
        }
    }
}


int
emtrees_bench_run(Emtrees *trees, EmtreesValue *values, int rows, int features, int repetitions) {

    int32_t class_sum = 0; // do something with data, prevents dead-code opt

    for (int r=0; r<repetitions; r++) {
        for (int row=0; row<rows; row++) {
            EmtreesValue *vals = values + (row * features); 
            const int32_t class_ = emtrees_predict(trees, vals, features);
            class_sum += class_;
        }
    }

    return class_sum;
}


#endif
