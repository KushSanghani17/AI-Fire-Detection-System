#include <WiFi.h>
#include "DHT.h"

#define DHT_PIN 4
#define DHT_TYPE DHT22

#define GAS_PIN 34
#define FLAME_PIN 27

// ===============================
// Wi-Fi Configuration
// ===============================
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";

// ===============================
// ThingSpeak Configuration
// ===============================
const char* THINGSPEAK_API_KEY = "THINGSPEAK_WRITE_API_KEY";
const char* THINGSPEAK_SERVER = "api.thingspeak.com";

DHT dht(DHT_PIN, DHT_TYPE);

WiFiClient client;


// ===============================
// Connect to Wi-Fi
// ===============================
void connectToWiFi() {

  Serial.println();
  Serial.println("Connecting to Wi-Fi...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}


// ===============================
// Setup
// ===============================
void setup() {

  Serial.begin(115200);

  dht.begin();

  pinMode(GAS_PIN, INPUT);
  pinMode(FLAME_PIN, INPUT_PULLUP);

  Serial.println("================================");
  Serial.println("   AI FIRE DETECTION SYSTEM");
  Serial.println("================================");

  connectToWiFi();
}


// ===============================
// Main Loop
// ===============================
void loop() {

  // =================================
  // Read DHT22
  // =================================

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Check DHT reading
  if (isnan(temperature) || isnan(humidity)) {

    Serial.println("ERROR: Failed to read DHT22.");

    delay(2000);

    return;
  }


  // =================================
  // Read MQ2 Gas Sensor
  // =================================

  int gasValue = analogRead(GAS_PIN);


  // =================================
  // Read Flame Simulation Button
  // =================================

  int flameState = digitalRead(FLAME_PIN);

  // INPUT_PULLUP:
  // LOW  = button pressed = flame detected
  // HIGH = button released = no flame

  int flameDetected = (flameState == LOW) ? 1 : 0;


  // =================================
  // Print Sensor Data
  // =================================

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


  // =================================
  // Send Data to ThingSpeak
  // =================================

  if (WiFi.status() == WL_CONNECTED) {

    if (client.connect(THINGSPEAK_SERVER, 80)) {

      String url = "/update?api_key=";

      url += THINGSPEAK_API_KEY;

      // Field 1 = Temperature
      url += "&field1=";
      url += String(temperature);

      // Field 2 = Humidity
      url += "&field2=";
      url += String(humidity);

      // Field 3 = Gas
      url += "&field3=";
      url += String(gasValue);

      // Field 4 = Flame
      url += "&field4=";
      url += String(flameDetected);


      client.print(
        String("GET ") + url + " HTTP/1.1\r\n" +
        "Host: " + THINGSPEAK_SERVER + "\r\n" +
        "Connection: close\r\n\r\n"
      );


      Serial.println();
      Serial.println("Data sent to ThingSpeak.");


      // =================================
      // Read ThingSpeak Response
      // =================================

      while (client.connected() || client.available()) {

        if (client.available()) {

          String line = client.readStringUntil('\n');

          Serial.println(line);
        }
      }

      client.stop();

    } else {

      Serial.println(
        "ERROR: Could not connect to ThingSpeak."
      );
    }

  } else {

    Serial.println(
      "ERROR: Wi-Fi disconnected."
    );
  }


  // =================================
  // ThingSpeak Update Interval
  // =================================

  delay(20000);
}
