from pathlib import Path
import os
import pickle
import requests
import pandas as pd


# ============================================================
# AI FIRE DETECTION - THINGSPEAK INTEGRATION
# ============================================================

AI_DIR = Path(__file__).resolve().parent

MODEL_PATH = AI_DIR / "model" / "random_forest.pkl"

# ============================================================
# THINGSPEAK CONFIGURATION
# ============================================================

THINGSPEAK_CHANNEL_ID = "3519763"

# Read API Key
# It will be taken from the environment variable.
THINGSPEAK_READ_API_KEY = os.getenv(
    "THINGSPEAK_READ_API_KEY",
    ""
)

# Write API Key
# It will be taken from the environment variable.
THINGSPEAK_WRITE_API_KEY = os.getenv(
    "THINGSPEAK_WRITE_API_KEY",
    ""
)

THINGSPEAK_READ_URL = (
    f"https://api.thingspeak.com/channels/"
    f"{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
)

THINGSPEAK_WRITE_URL = (
    "https://api.thingspeak.com/update"
)


# ============================================================
# 1. LOAD RANDOM FOREST MODEL
# ============================================================

def load_model():

    print("\nLoading Random Forest model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nERROR: Model not found:\n{MODEL_PATH}\n"
            "\nPlease run train_rf.py first."
        )

    with open(MODEL_PATH, "rb") as file:

        model = pickle.load(file)

    print("Random Forest model loaded successfully.")

    return model


# ============================================================
# 2. GET LATEST DATA FROM THINGSPEAK
# ============================================================

def get_thingspeak_data():

    print("\nReading latest data from ThingSpeak...")

    # Check Read API Key
    if not THINGSPEAK_READ_API_KEY:

        raise ValueError(
            "\nERROR: ThingSpeak Read API Key is not configured.\n\n"
            "Set it in PowerShell using:\n"
            '$env:THINGSPEAK_READ_API_KEY="YOUR_READ_API_KEY"'
        )

    params = {
        "api_key": THINGSPEAK_READ_API_KEY
    }

    try:

        response = requests.get(
            THINGSPEAK_READ_URL,
            params=params,
            timeout=10
        )

        print(
            f"ThingSpeak HTTP Status: "
            f"{response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

        if not data:

            raise ValueError(
                "ThingSpeak returned empty data."
            )

        # ====================================================
        # ThingSpeak Fields
        # ====================================================
        # Field 1 = Temperature
        # Field 2 = Humidity
        # Field 3 = Gas
        # Field 4 = Flame

        if data.get("field1") is None:
            raise ValueError(
                "ThingSpeak Field 1 is empty."
            )

        if data.get("field2") is None:
            raise ValueError(
                "ThingSpeak Field 2 is empty."
            )

        if data.get("field3") is None:
            raise ValueError(
                "ThingSpeak Field 3 is empty."
            )

        if data.get("field4") is None:
            raise ValueError(
                "ThingSpeak Field 4 is empty."
            )

        temperature = float(
            data["field1"]
        )

        humidity = float(
            data["field2"]
        )

        gas = float(
            data["field3"]
        )

        flame = float(
            data["field4"]
        )

        print(
            "ThingSpeak data received successfully."
        )

        print("\nLatest Sensor Data:")

        print(
            f"Temperature : "
            f"{temperature:.2f} °C"
        )

        print(
            f"Humidity    : "
            f"{humidity:.2f} %"
        )

        print(
            f"Gas         : "
            f"{gas:.2f}"
        )

        print(
            "Flame       : "
            + (
                "DETECTED"
                if flame == 1
                else "NOT DETECTED"
            )
        )

        return (
            temperature,
            humidity,
            gas,
            flame
        )

    except requests.exceptions.RequestException as e:

        print(
            "\nERROR: Could not read data "
            "from ThingSpeak."
        )

        print(e)

        raise

    except (KeyError, TypeError, ValueError) as e:

        print(
            "\nERROR: Invalid ThingSpeak data."
        )

        print(e)

        raise


# ============================================================
# 3. RANDOM FOREST PREDICTION
# ============================================================

def predict_fire_status(
    model,
    temperature,
    humidity,
    gas,
    flame
):

    # Create DataFrame using the same
    # features used during training.

    sensor_data = pd.DataFrame({

        "temperature": [temperature],

        "humidity": [humidity],

        "gas": [gas],

        "flame": [flame]

    })

    # Make prediction

    prediction = model.predict(
        sensor_data
    )

    predicted_status = prediction[0]

    # Prediction probabilities

    probabilities = model.predict_proba(
        sensor_data
    )

    probability_data = dict(
        zip(
            model.classes_,
            probabilities[0]
        )
    )

    return (
        predicted_status,
        probability_data
    )


# ============================================================
# 4. SEND AI RESULT TO THINGSPEAK FIELD 5
# ============================================================

# ============================================================
# 4. SEND AI RESULT TO THINGSPEAK
# ============================================================

def send_prediction_to_thingspeak(predicted_status):

    if not THINGSPEAK_WRITE_API_KEY:
        print("\nWARNING: ThingSpeak Write API key not configured.")
        return

    print("\nSending AI prediction to ThingSpeak...")

    # Convert AI status to numeric value
    status_values = {
        "Normal": 0,
        "Warning": 1,
        "Critical": 2
    }

    status_value = status_values.get(predicted_status, -1)

    # Field 5 = Random Forest prediction
    # Field 7 = Final Status
    params = {
        "api_key": THINGSPEAK_WRITE_API_KEY,
        "field5": status_value,
        "field7": status_value
    }

    try:
        response = requests.get(
            "https://api.thingspeak.com/update",
            params=params,
            timeout=10
        )

        print(f"ThingSpeak Write HTTP Status: {response.status_code}")
        print(f"ThingSpeak Response: {response.text}")

        response.raise_for_status()

        if response.text == "0":

            print("\nThingSpeak update rejected.")
            print("Waiting 16 seconds and retrying...")

            import time
            time.sleep(16)

            response = requests.get(
                "https://api.thingspeak.com/update",
                params=params,
                timeout=10
            )

            print(
                f"Retry HTTP Status: {response.status_code}"
            )
            print(
                f"Retry Response: {response.text}"
            )

            response.raise_for_status()

            if response.text == "0":
                print(
                    "ERROR: ThingSpeak rejected the retry."
                )
            else:
                print(
                    "AI prediction uploaded successfully "
                    "after retry."
                )
                print(
                    f"ThingSpeak entry ID: {response.text}"
                )

        else:

            print(
                "AI prediction uploaded successfully."
            )

            print(
                f"ThingSpeak entry ID: {response.text}"
            )

    except requests.exceptions.RequestException as e:

        print(
            "\nERROR: Could not write to ThingSpeak."
        )
        print(e)


# ============================================================
# 5. DISPLAY FINAL RESULT
# ============================================================

def display_result(
    temperature,
    humidity,
    gas,
    flame,
    predicted_status,
    probabilities
):

    print("\n" + "=" * 60)

    print(
        "              SENSOR DATA"
    )

    print("=" * 60)

    print(
        f"Temperature : "
        f"{temperature:.2f} °C"
    )

    print(
        f"Humidity    : "
        f"{humidity:.2f} %"
    )

    print(
        f"Gas Value   : "
        f"{gas:.2f}"
    )

    print(
        "Flame       : "
        + (
            "DETECTED"
            if flame == 1
            else "NOT DETECTED"
        )
    )

    print("\n" + "=" * 60)

    print(
        "              AI PREDICTION"
    )

    print("=" * 60)

    print(
        f"\nPredicted Status: "
        f"{predicted_status}"
    )

    print(
        "\nPrediction Probabilities:"
    )

    for class_name, probability in (
        probabilities.items()
    ):

        print(
            f"  {class_name:10s}: "
            f"{probability * 100:.2f}%"
        )

    print("\n" + "=" * 60)

    if predicted_status == "Critical":

        print(
            "!!! CRITICAL FIRE ALERT !!!"
        )

    elif predicted_status == "Warning":

        print(
            "!! FIRE WARNING !!"
        )

    else:

        print(
            "SYSTEM STATUS: NORMAL"
        )

    print("=" * 60)


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "       AI FIRE DETECTION SYSTEM"
    )

    print(
        "       THINGSPEAK + RANDOM FOREST"
    )

    print("=" * 60)

    # ========================================================
    # Step 1: Load trained AI model
    # ========================================================

    model = load_model()

    # ========================================================
    # Step 2: Read latest ESP32 data
    # from ThingSpeak
    # ========================================================

    (
        temperature,
        humidity,
        gas,
        flame
    ) = get_thingspeak_data()

    # ========================================================
    # Step 3: AI prediction
    # ========================================================

    (
        predicted_status,
        probabilities
    ) = predict_fire_status(
        model,
        temperature,
        humidity,
        gas,
        flame
    )

    # ========================================================
    # Step 4: Display result
    # ========================================================

    display_result(
        temperature,
        humidity,
        gas,
        flame,
        predicted_status,
        probabilities
    )

    # ========================================================
    # Step 5: Send AI result to ThingSpeak
    # Field 5
    # ========================================================

    send_prediction_to_thingspeak(
        predicted_status
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()