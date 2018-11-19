
#include <eml_trees.h>
#include <eml_benchmark.h>

#include "digits.h"


void send_reply(int32_t request, int32_t time_taken,
              int32_t prediction, int32_t n_repetitions)
{
    Serial.print(request);
    Serial.print(";");
    Serial.print(time_taken);
    Serial.print(";");
    Serial.print(prediction);
    //Serial.print(";");
    //Serial.print(n_repetitions);
    //for (int i=0; i<n_features; i++) {
    //  Serial.print(";");
    //  Serial.print(values[i]);
    //}
    Serial.print("\n");
}

void parse_predict_reply(char *buffer, float *values, int32_t values_length)
{
  // FIXME: buffer needs to be zero terminated?

  // Parse the values to use for prediction
  int32_t n_values = -1;
  const EmlError e = eml_test_parse_csv_line(buffer, values, values_length, &n_values);
  if (e != EmlOk) {
      return;
  }

  const expected_values = n_features + 2;
  if (n_values != n_features) {
      return;
  }        

  int32_t request = values[0];
  int32_t n_repetitions = values[1];
  values = values+2;
  n_values -= 2;

  // Do predictions
  volatile int32_t sum = 0; // avoid profiler folding the loop
  const long pre = micros();

  int32_t prediction = -999;
  for (int32_t i=0; i<n_repetitions; i++) {
    const int32_t p = digits_predict(values, n_values);
    if (prediction != -999 && p != prediction) {
        // consistency check, should always be same
        prediction = -2;
        break;
    }
    //Serial.print("cl: "); Serial.println(s);
    sum += p;
    prediction = p;
  }

  const long post = micros();
  const long time_taken = post - pre;     

  // Send back on parseable format
  send_reply(request, time_taken, prediction, n_repetitions);
}


void setup() {
  Serial.begin(115200);
}

void loop() {
  const int32_t n_features = 64;
  const int32_t bytes_per_number = 20; // 32bit integer, plus separator. But better to have too much
  const int32_t buffer_length = n_features*bytes_per_number;

  char receive_buffer[buffer_length] = {0,};
  const int32_t values_length = 4+n_features;
  float values[values_length];
  int32_t receive_idx = 0;

  while (Serial.available() > 0) {

    const char ch = Serial.read();
    receive_buffer[receive_idx++] = ch;

    if (receive_idx >= buffer_length-1) {
        receive_idx = 0;
        memset(receive_buffer, buffer_length, 0);
        Serial.println("Error, buffer overflow");
    }
    
    if (ch == '\n') {
        parse_predict_reply();

        receive_idx = 0;
        memset(receive_buffer, buffer_length, 0);
    }
  }
}
