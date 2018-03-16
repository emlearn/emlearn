
#include "emtrees_bench.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mytree.h"

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); 
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000;
    return milliseconds;
}

int main() {

    const int n_rows = 1000;
    const int n_features = 61;
    const int n_repetitions = 1000;
    Emtrees *trees = &myclassifier;
    EmtreesValue *values = (EmtreesValue *)malloc(n_rows*n_features*sizeof(EmtreesValue));

    emtrees_bench_fill(values, n_rows, n_features);
    
    printf("%d trees, %d nodes\n", trees->n_trees, trees->n_nodes);
    printf("%d repetitions of %d rows with %d features\n",
            n_repetitions, n_rows, n_features);

    const long long pre = current_timestamp();
    const int32_t s = emtrees_bench_run(trees, values, n_rows, n_features, n_repetitions);
    const long long post = current_timestamp();
    const long time_taken = post - pre; 
    const long n_classifications = n_rows * n_repetitions;    

    printf("took %ld milliseconds\n", time_taken);
    printf("%ld classifications/msec\n", n_classifications / time_taken);
    printf("class sum: %d\n", s);

    free(values);
}
