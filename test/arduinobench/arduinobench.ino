
#include "emtrees.h"
#include "emtrees_bench.h"
#include "mytree.h"

const int32_t n_rows = 5;
const int32_t n_features = 61;
const int n_repetitions = 500;
EmtreesValue values[n_rows*n_features];
Emtrees *trees = &myclassifier;

void setup() {
  Serial.begin(115200); 
  Serial.println("running fill");
  emtrees_bench_fill(values, n_rows, n_features);
  Serial.println("fill done");
}

void loop() {

  //Serial.print("%d trees, %d nodes\n", trees->n_trees, trees->n_nodes);
  //Serial.printf("%d repetitions of %d rows with %d features\n",
  //          n_repetitions, n_rows, n_features);

  const long pre = millis();
  const int32_t s = emtrees_bench_run(trees, values, n_rows, n_features, n_repetitions);
  const long post = millis();
  const long time_taken = post - pre; 
  const long n_classifications = n_rows * n_repetitions;    

  Serial.print(time_taken); Serial.println("milliseconds");
  Serial.print(n_classifications / time_taken); Serial.println("classifications/msec");
  delay(1000);
  //printf("class sum: %d\n", s);
}
