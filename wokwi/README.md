# Wokwi ESP32 Fire Detection Simulation

## 1. Overview

This folder contains the Wokwi simulation for the IoT-based AI Fire Detection System.

The Wokwi simulation is used to emulate the sensor layer of the proposed fire detection system before integrating the sensor data with ThingSpeak and the AI-based detection modules.

The ESP32 collects environmental data from:

- DHT22 temperature and humidity sensor
- MQ2 gas sensor
- Pushbutton used as a simulated flame input

The collected sensor values are later intended to be used for cloud monitoring through ThingSpeak and for AI-based fire detection and decision fusion.

---

## 2. System Components

### ESP32 DevKit

The ESP32 is the main microcontroller of the simulation.

It is responsible for:

- Reading temperature and humidity
- Reading gas sensor values
- Reading the simulated flame input
- Processing sensor readings
- Preparing sensor data for further cloud and AI integration

---

### DHT22 Sensor

The DHT22 sensor is used to measure:

- Temperature
- Humidity

The temperature and humidity readings provide environmental information that can be used as inputs for fire-risk analysis.

### DHT22 Pin Configuration

| DHT22 Pin | ESP32 Pin |
|-----------|-----------|
| VCC | 3V3 |
| SDA / DATA | GPIO 4 |
| NC | Not Connected |
| GND | GND |

---

## 3. MQ2 Gas Sensor

The MQ2 gas sensor is used to simulate gas/smoke level measurements.

The analog output of the sensor is connected to an ESP32 ADC pin so that a numerical gas value can be obtained.

### MQ2 Pin Configuration

| MQ2 Pin | ESP32 Pin |
|---------|-----------|
| VCC | 3V3 |
| AOUT | GPIO 34 |
| DOUT | Not Connected |
| GND | GND |

Only the analog output (AOUT) is used because the project requires a numerical gas/smoke reading.

---

## 4. Simulated Flame Sensor

A physical flame sensor is not used in the Wokwi simulation.

Instead, a pushbutton is used as a static digital flame input.

This allows the fire condition to be manually simulated during testing and demonstration.

### Pushbutton Logic

```text
Button Released → Flame = 0 → No Flame
Button Pressed  → Flame = 1 → Flame Detected
