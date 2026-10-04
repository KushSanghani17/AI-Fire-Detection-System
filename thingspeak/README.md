# ThingSpeak Cloud Integration

## 1. Overview

The ThingSpeak module provides cloud-based monitoring for the AI Fire Detection System.

The ESP32 collects environmental sensor data and sends the data to ThingSpeak through Wi-Fi.

The current implementation sends:

- Temperature
- Humidity
- Gas level
- Flame status

The ThingSpeak channel stores this information and displays it using live graphs.

---

## 2. Data Flow

The current data flow is:

ESP32 Sensors
↓
ESP32
↓
Wi-Fi
↓
ThingSpeak
↓
Cloud Channel
↓
Graphs and Monitoring

---

## 3. ThingSpeak Channel

The project uses the following ThingSpeak channel fields:

| Field | Parameter | Description |
|---|---|---|
| Field 1 | Temperature | Temperature measured by DHT22 |
| Field 2 | Humidity | Relative humidity measured by DHT22 |
| Field 3 | Gas | Analog gas sensor reading from MQ2 |
| Field 4 | Flame | Simulated flame detection status |
| Field 5 | Random Forest | Machine learning classification |
| Field 6 | YOLO | Computer vision fire detection |
| Field 7 | Final Status | Final fused fire detection decision |

Fields 5, 6, and 7 will be integrated after the AI and computer vision modules are implemented.

---

## 4. Current Sensor Integration

### Temperature

The DHT22 sensor provides the temperature value.

ESP32 pin:

GPIO 4

ThingSpeak:

Field 1

---

### Humidity

The DHT22 sensor also provides relative humidity.

ESP32 pin:

GPIO 4

ThingSpeak:

Field 2

---

### Gas

The MQ2 gas sensor provides an analog gas reading.

ESP32 pin:

GPIO 34

ThingSpeak:

Field 3

---

### Flame

A pushbutton is used as a simulated flame input because a physical flame sensor is not available in the simulation.

ESP32 pin:

GPIO 27

The button uses `INPUT_PULLUP`.

Therefore:

- Button released → Flame not detected → 0
- Button pressed → Flame detected → 1

ThingSpeak:

Field 4

---

## 5. Update Process

The ESP32 connects to the Wokwi Wi-Fi network and sends sensor values to ThingSpeak.

The data is transmitted using the ThingSpeak HTTP update API.

The current update interval is approximately 20 seconds.

This interval prevents excessively frequent updates and is suitable for the current simulation.

---

## 6. Example Data

Example sensor data sent to ThingSpeak:

Temperature:

25.5 °C

Humidity:

60 %

Gas:

1200

Flame:

0

This information is displayed in the ThingSpeak channel graphs.

---

## 7. Current Implementation Status

| Component | Status |
|---|---|
| ESP32 | Completed |
| DHT22 | Completed |
| MQ2 | Completed |
| Flame simulation | Completed |
| Wi-Fi connection | Completed |
| ThingSpeak channel | Completed |
| Sensor data upload | Completed |
| ThingSpeak graphs | Completed |
| Random Forest integration | Pending |
| YOLO integration | Pending |
| Final decision fusion | Pending |

---

## 8. Security

The ThingSpeak Write API Key must not be committed to a public GitHub repository.

API keys should be stored locally or through a secure configuration mechanism.

The repository should contain only example configuration files and never expose the actual API key.

---

## 9. Future Integration

After the sensor-to-cloud communication is completed, the following modules will be connected:

Random Forest
↓
Environmental classification

YOLOv8
↓
Visual fire detection

Temporal Confirmation
↓
Multi-frame verification

Decision Fusion
↓
Final fire decision

The final result will eventually be sent back to ThingSpeak using:

- Field 5 → Random Forest
- Field 6 → YOLO
- Field 7 → Final Status
