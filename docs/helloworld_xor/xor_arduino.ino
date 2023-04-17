
#include "xor_model.h"

const int ledPin = LED_BUILTIN;
const int digitalPinA = 8;
const int digitalPinB = 7;
const int analogPinA = 4;
const int analogPinB = 5;


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
  // put your main code here, to run repeatedly:

  const float a = readDigital(digitalPinA);
  const float b = readDigital(digitalPinB);
  const float features[] = { a, b };

  const int32_t out = xor_predict(features, 2);
  if (out < 0) {
    Serial.println("Error");
  }
  digitalWrite(ledPin, out == 1 ? HIGH : LOW);
  delay(100);
}
