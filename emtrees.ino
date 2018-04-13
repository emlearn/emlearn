
#include "emtrees.h"
#include "emtrees_bench.h"
#include "digits.h"

const int32_t n_features = 64;

const int32_t bytes_per_number = 50; // 32bit integer, plus separator. But better to have too much
const int32_t buffer_length = n_features*bytes_per_number;
char receive_buffer[buffer_length] = {0,};
int32_t receive_idx = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  EmtreesValue values[n_features];

  while (Serial.available() > 0) {

    const char ch = Serial.read();
    receive_buffer[receive_idx++] = ch;

    if (receive_idx >= buffer_length-1) {
        receive_idx = 0;
        memset(receive_buffer, buffer_length, 0);
        Serial.println("Error, buffer overflow");
    }
    
    if (ch == '\n') {
        int32_t request = -3;
        int32_t n_repetitions = -1;
        const int non_value_fields = 2;
        
        // Parse the values to use for prediction
        int field_no = 0;
        char seps[] = ",;";
        char *token = strtok(receive_buffer, seps);
        while (token != NULL)
        {
            long value;
            sscanf(token, "%ld", &value);
            if (field_no == 0) {
              request = value;
            } else if (field_no == 1) {
              n_repetitions = value;
            } else {
              values[field_no-non_value_fields] = value; 
            }
            field_no++;
            token = strtok(NULL, seps);
        }

        if (field_no-non_value_fields != n_features) {
            Serial.print("Error, wrong number of features: "); Serial.println(field_no-non_value_fields);
        }
        
        receive_idx = 0;
        memset(receive_buffer, buffer_length, 0);

        // Do predictions
        volatile int32_t sum = 0; // avoid profiler folding the loop
        const long pre = micros();
        int32_t prediction = -999;
        for (int32_t i=0; i<n_repetitions; i++) {
          const int32_t p = digits_predict(values, n_features);
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
        Serial.print(request);
        Serial.print(";");
        Serial.print(time_taken);
        Serial.print(";");
        Serial.print(prediction);
        Serial.print(";");
        Serial.print(n_repetitions);
        for (int i=0; i<n_features; i++) {
          Serial.print(";");
          Serial.print(values[i]);
        }
        Serial.print("\n");
    }
  }
}
