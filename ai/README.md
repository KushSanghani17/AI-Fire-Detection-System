# Artificial Intelligence Module

## 1. Overview

The Artificial Intelligence (AI) module is responsible for analyzing
environmental sensor data and classifying the current fire-risk condition.

The system uses a Random Forest machine learning algorithm to analyze
sensor values received from the fire detection system.

The main purpose of this module is to convert raw sensor readings into
an understandable fire-risk classification.

The classification output is:

- Normal
- Warning
- Critical

---

## 2. Role of the AI Module

The AI module acts as the environmental intelligence layer of the
AI-Based Fire Detection and Emergency Alert System.

The sensor layer collects raw data such as:

- Temperature
- Humidity
- Gas concentration/value
- Flame status

The AI model analyzes these values and determines the current
environmental condition.

---

## 3. AI Data Flow

The AI processing pipeline is:

Sensor Data
↓
Data Collection
↓
Data Preprocessing
↓
Feature Extraction
↓
Random Forest Model
↓
Fire-Risk Classification
↓
Normal / Warning / Critical

---

## 4. Input Features

The Random Forest model uses the following features:

| Feature | Description |
|---|---|
| Temperature | Temperature measured by the DHT22 sensor |
| Humidity | Relative humidity measured by the DHT22 sensor |
| Gas | Analog gas sensor reading from the MQ2 sensor |
| Flame | Flame detection status from the simulated flame input |

These values are used as input features for the machine learning model.

---

## 5. Output Classes

The model produces one of three fire-risk classifications.

### Normal

The environmental conditions do not indicate a significant fire
risk.

Example:

- Normal temperature
- Normal gas level
- No flame detected

Output:

```text
NORMAL
