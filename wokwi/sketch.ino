#include "DHT.h"

#define DHT_PIN 4
#define DHT_TYPE DHT22

#define GAS_PIN 34
#define FLAME_PIN 27

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);

  dht.begin();

  pinMode(GAS_PIN, INPUT);
  pinMode(FLAME_PIN, INPUT_PULLUP);

  Serial.println("================================");
  Serial.println("   FIRE DETECTION SYSTEM");
  Serial.println("================================");
}

void loop() {

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int gasValue = analogRead(GAS_PIN);

  int flameState = digitalRead(FLAME_PIN);

  int flameDetected = (flameState == LOW) ? 1 : 0;

  Serial.println("--------------------------------");

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print("Gas Value: ");
  Serial.println(gasValue);

  Serial.print("Flame: ");

  if (flameDetected == 1) {
    Serial.println("DETECTED");
  } else {
    Serial.println("NOT DETECTED");
  }

  delay(2000);
}
