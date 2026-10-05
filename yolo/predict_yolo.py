from pathlib import Path
from ultralytics import YOLO


# ============================================================
# YOLO FIRE + SMOKE PREDICTION
# ============================================================

YOLO_DIR = Path(__file__).resolve().parent

# Trained YOLO model
MODEL_PATH = (
    YOLO_DIR
    / "runs"
    / "fire_smoke"
    / "weights"
    / "best.pt"
)

# Test image
IMAGE_PATH = YOLO_DIR / "test_image.jpg"

# Detection confidence
CONFIDENCE = 0.25


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("       YOLO FIRE + SMOKE DETECTION")
    print("=" * 70)

    print("\nLoading trained YOLO model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nERROR: best.pt not found:\n{MODEL_PATH}\n"
            "\nMake sure YOLO training is completed."
        )

    model = YOLO(str(MODEL_PATH))

    print("YOLO model loaded successfully.")

    return model


# ============================================================
# 2. RUN PREDICTION
# ============================================================

def predict_image(model):

    print("\nReading test image...")

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"\nERROR: Test image not found:\n{IMAGE_PATH}\n"
            "\nPut an image named 'test_image.jpg' inside:\n"
            f"{YOLO_DIR}"
        )

    print(f"Image: {IMAGE_PATH}")

    print("\nRunning YOLO detection...")

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=CONFIDENCE,
        save=True,
        project=str(YOLO_DIR / "runs"),
        name="prediction",
        exist_ok=True,
        verbose=False,
    )

    return results


# ============================================================
# 3. DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n" + "=" * 70)
    print("                 DETECTION RESULT")
    print("=" * 70)

    detection_found = False

    for result in results:

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:

            print("\nNo fire or smoke detected.")

            continue

        detection_found = True

        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            print(
                f"\nDetected : {class_name.upper()}"
            )

            print(
                f"Confidence : {confidence * 100:.2f}%"
            )

            if class_name.lower() == "fire":

                print("STATUS : FIRE DETECTED")

            elif class_name.lower() == "smoke":

                print("STATUS : SMOKE DETECTED")

    print("\n" + "=" * 70)

    if detection_found:

        print("YOLO DETECTION COMPLETED")

    else:

        print("SYSTEM STATUS: NORMAL")

    print("=" * 70)

    print(
        "\nResult image saved inside:"
    )

    print(
        YOLO_DIR
        / "runs"
        / "prediction"
    )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    model = load_model()

    results = predict_image(model)

    display_results(results)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()