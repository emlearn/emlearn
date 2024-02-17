
#include "xor_model.h"

const int ledPin = LED_BUILTIN;
const int digitalPinA = 9;
const int digitalPinB = 8;
const int analogPinA = A1;
const int analogPinB = A2;


void setup() {
  pinMode(ledPin, OUTPUT); 
  pinMode(digitalPinA, INPUT_PULLUP); 
  pinMode(digitalPinB, INPUT_PULLUP); 
}

// Read values into range 0.0-1.0
float readAnalog(const int pin) {
    return analogRead(pin) / 1023.0;
}
float readDigital(const int pin) {
    return digitalRead(pin) ? 1.0 : 0.0;
}


void loop() {

#if 1
  // use digital pins
  const float a = readDigital(digitalPinA);
  const float b = readDigital(digitalPinB);
#else
  // use analog pins
  const float a = readAnalog(analogPinA);
  const float b = readAnalog(analogPinB);
#endif
  const float features[] = { a, b };

  const int32_t out = xor_model_predict(features, 2);

  if (out < 0) {
    Serial.println("ERROR");
  } else {
    Serial.print("Output class:");
    Serial.println(out);
  }
  digitalWrite(ledPin, out == 1 ? HIGH : LOW);
  delay(100);
}
