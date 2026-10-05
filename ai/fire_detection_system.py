from pathlib import Path
import sys
import os
import requests

from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

AI_DIR = Path(__file__).resolve().parent

# Make sure Python can find predict.py
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))


# ============================================================
# IMPORT RANDOM FOREST FUNCTIONS
# ============================================================

from predict import (
    load_model,
    get_thingspeak_data,
    predict_fire_status
)


# ============================================================
# AI FIRE DETECTION SYSTEM
# RANDOM FOREST + THINGSPEAK + YOLO
# ============================================================


# ============================================================
# YOLO CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = (

    AI_DIR

    / "yolo"

    / "runs"

    / "fire_smoke"

    / "weights"

    / "best.pt"

)

YOLO_IMAGE_PATH = (

    AI_DIR

    / "yolo"

    / "test_image.jpg"

)


# ============================================================
# THINGSPEAK CONFIGURATION
# ============================================================

THINGSPEAK_CHANNEL_ID = "3519763"

THINGSPEAK_WRITE_API_KEY = os.getenv(

    "THINGSPEAK_WRITE_API_KEY",

    ""

)

THINGSPEAK_WRITE_URL = (

    "https://api.thingspeak.com/update"

)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

def load_yolo_model():

    print(
        "\nLoading YOLO fire detection model..."
    )

    print(
        "YOLO model path:"
    )

    print(
        YOLO_MODEL_PATH
    )

    if not YOLO_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"\nYOLO model not found:\n"
            f"{YOLO_MODEL_PATH}"

        )

    model = YOLO(

        str(
            YOLO_MODEL_PATH
        )

    )

    print(
        "YOLO model loaded successfully."
    )

    return model


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_with_yolo(model):

    print(
        "\nRunning YOLO fire detection..."
    )

    print(
        "Image:"
    )

    print(
        YOLO_IMAGE_PATH
    )

    if not YOLO_IMAGE_PATH.exists():

        raise FileNotFoundError(

            f"\nTest image not found:\n"
            f"{YOLO_IMAGE_PATH}"

        )

    results = model.predict(

        source=str(
            YOLO_IMAGE_PATH
        ),

        conf=0.25,

        save=True,

        project=str(
            AI_DIR / "yolo" / "runs"
        ),

        name="integration_prediction",

        exist_ok=True

    )

    detections = []

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            class_id = int(

                box.cls[0].item()

            )

            confidence = float(

                box.conf[0].item()

            )

            class_name = model.names[

                class_id

            ]

            class_name = str(

                class_name

            ).lower()

            print(

                f"YOLO Detection: "
                f"{class_name} "
                f"{confidence * 100:.2f}%"

            )

            detections.append({

                "class":
                    class_name,

                "confidence":
                    confidence

            })

    return detections


# ============================================================
# ANALYZE YOLO RESULT
# ============================================================

def analyze_yolo_result(detections):

    fire_detected = False

    smoke_detected = False

    highest_fire_confidence = 0.0

    highest_smoke_confidence = 0.0

    for detection in detections:

        class_name = detection["class"]

        confidence = detection["confidence"]

        if class_name == "fire":

            fire_detected = True

            highest_fire_confidence = max(

                highest_fire_confidence,

                confidence

            )

        elif class_name == "smoke":

            smoke_detected = True

            highest_smoke_confidence = max(

                highest_smoke_confidence,

                confidence

            )

    return (

        fire_detected,

        smoke_detected,

        highest_fire_confidence,

        highest_smoke_confidence

    )


# ============================================================
# FINAL DECISION
# ============================================================

def get_final_status(

    rf_status,

    fire_detected,

    smoke_detected

):

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if rf_status == "Critical":

        return "Critical"

    if fire_detected:

        return "Critical"

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    if rf_status == "Warning":

        return "Warning"

    if smoke_detected:

        return "Warning"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "Normal"


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_result(

    temperature,

    humidity,

    gas,

    flame,

    rf_status,

    rf_probabilities,

    fire_detected,

    smoke_detected,

    fire_confidence,

    smoke_confidence,

    final_status

):

    print("\n")

    print("=" * 65)

    print(
        "                 FINAL FIRE DETECTION"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # SENSOR DATA
    # --------------------------------------------------------

    print("\nSENSOR DATA")

    print("-" * 65)

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

        +

        (

            "DETECTED"

            if flame == 1

            else "NOT DETECTED"

        )

    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\nRANDOM FOREST")

    print("-" * 65)

    print(

        f"RF Prediction : "
        f"{rf_status}"

    )

    print(
        "\nRF Probabilities:"
    )

    for (

        class_name,

        probability

    ) in rf_probabilities.items():

        print(

            f"  {class_name:10s}: "
            f"{probability * 100:.2f}%"

        )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("\nYOLO")

    print("-" * 65)

    print(

        "Fire Detected  : "

        +

        (

            "YES"

            if fire_detected

            else "NO"

        )

    )

    print(

        "Smoke Detected : "

        +

        (

            "YES"

            if smoke_detected

            else "NO"

        )

    )

    print(

        f"Fire Confidence  : "
        f"{fire_confidence * 100:.2f}%"

    )

    print(

        f"Smoke Confidence : "
        f"{smoke_confidence * 100:.2f}%"

    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "                    FINAL STATUS"
    )

    print("=" * 65)

    print(

        f"\n                 "
        f"{final_status.upper()}"

    )

    if final_status == "Critical":

        print(
            "\n!!! CRITICAL FIRE ALERT !!!"
        )

    elif final_status == "Warning":

        print(
            "\n!! FIRE WARNING !!"
        )

    else:

        print(
            "\nSYSTEM STATUS: NORMAL"
        )

    print("=" * 65)


# ============================================================
# SEND COMPLETE RESULT TO THINGSPEAK
# ============================================================

def send_final_status_to_thingspeak(

    temperature,

    humidity,

    gas,

    flame,

    rf_status,

    fire_detected,

    smoke_detected,

    final_status

):

    if not THINGSPEAK_WRITE_API_KEY:

        print(

            "\nWARNING: THINGSPEAK_WRITE_API_KEY "
            "is not configured."

        )

        return

    # --------------------------------------------------------
    # STATUS VALUES
    # --------------------------------------------------------

    status_values = {

        "Normal": 0,

        "Warning": 1,

        "Critical": 2

    }

    rf_value = status_values.get(

        rf_status,

        -1

    )

    final_value = status_values.get(

        final_status,

        -1

    )

    # --------------------------------------------------------
    # YOLO VALUE
    # --------------------------------------------------------

    if fire_detected:

        yolo_value = 2

    elif smoke_detected:

        yolo_value = 1

    else:

        yolo_value = 0

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nSending complete AI result "
        "to ThingSpeak..."
    )

    print(
        f"Temperature : {temperature}"
    )

    print(
        f"Humidity    : {humidity}"
    )

    print(
        f"Gas         : {gas}"
    )

    print(
        f"Flame       : {flame}"
    )

    print(
        f"RF          : "
        f"{rf_status} ({rf_value})"
    )

    print(
        f"YOLO        : "
        f"{yolo_value}"
    )

    print(
        f"Final       : "
        f"{final_status} ({final_value})"
    )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    params = {

        "api_key":
            THINGSPEAK_WRITE_API_KEY,

        "field1":
            temperature,

        "field2":
            humidity,

        "field3":
            gas,

        "field4":
            flame,

        "field5":
            rf_value,

        "field6":
            yolo_value,

        "field7":
            final_value

    }

    try:

        response = requests.get(

            THINGSPEAK_WRITE_URL,

            params=params,

            timeout=10

        )

        print(

            f"\nThingSpeak HTTP Status: "
            f"{response.status_code}"

        )

        print(

            f"ThingSpeak Response: "
            f"{response.text}"

        )

        response.raise_for_status()

        if response.text.strip() == "0":

            print(

                "\nThingSpeak rejected "
                "the update."

            )

            print(

                "Possible reason: "
                "ThingSpeak update rate limit."

            )

        else:

            print(

                "\nComplete AI result "
                "uploaded successfully."

            )

            print(

                f"ThingSpeak Entry ID: "
                f"{response.text}"

            )

    except requests.exceptions.RequestException as e:

        print(

            "\nERROR: Could not write "
            "results to ThingSpeak."

        )

        print(e)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)

    print(
        "          AI FIRE DETECTION SYSTEM"
    )

    print(
        "     YOLO + RANDOM FOREST + THINGSPEAK"
    )

    print("=" * 65)

    # ========================================================
    # STEP 1
    # RANDOM FOREST
    # ========================================================

    print(
        "\nLoading Random Forest model..."
    )

    rf_model = load_model()

    print(
        "Random Forest model loaded."
    )

    # ========================================================
    # STEP 2
    # SENSOR DATA
    # ========================================================

    print(
        "\nGetting sensor data from ThingSpeak..."
    )

    (
        temperature,
        humidity,
        gas,
        flame

    ) = get_thingspeak_data()

    print(
        "\nSensor data received."
    )

    print(
        f"Temperature: {temperature}"
    )

    print(
        f"Humidity: {humidity}"
    )

    print(
        f"Gas: {gas}"
    )

    print(
        f"Flame: {flame}"
    )

    # ========================================================
    # STEP 3
    # RANDOM FOREST PREDICTION
    # ========================================================

    print(
        "\nRunning Random Forest..."
    )

    (
        rf_status,

        rf_probabilities

    ) = predict_fire_status(

        rf_model,

        temperature,

        humidity,

        gas,

        flame

    )

    print(
        "\nRF Prediction:",
        rf_status
    )

    # ========================================================
    # STEP 4
    # LOAD YOLO
    # ========================================================

    yolo_model = load_yolo_model()

    # ========================================================
    # STEP 5
    # RUN YOLO
    # ========================================================

    detections = detect_with_yolo(

        yolo_model

    )

    # ========================================================
    # STEP 6
    # ANALYZE YOLO
    # ========================================================

    (

        fire_detected,

        smoke_detected,

        fire_confidence,

        smoke_confidence

    ) = analyze_yolo_result(

        detections

    )

    # ========================================================
    # STEP 7
    # FINAL DECISION
    # ========================================================

    final_status = get_final_status(

        rf_status,

        fire_detected,

        smoke_detected

    )

    # ========================================================
    # STEP 8
    # DISPLAY
    # ========================================================

    display_final_result(

        temperature,

        humidity,

        gas,

        flame,

        rf_status,

        rf_probabilities,

        fire_detected,

        smoke_detected,

        fire_confidence,

        smoke_confidence,

        final_status

    )

    # ========================================================
    # STEP 9
    # SEND TO THINGSPEAK
    # ========================================================

    send_final_status_to_thingspeak(

        temperature,

        humidity,

        gas,

        flame,

        rf_status,

        fire_detected,

        smoke_detected,

        final_status

    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()