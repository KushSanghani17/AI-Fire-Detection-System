from pathlib import Path
import pickle
import pandas as pd


# ============================================================
# FIRE DETECTION - PREDICTION PROGRAM
# ============================================================

# predict.py is inside the ai/ folder
AI_DIR = Path(__file__).resolve().parent

MODEL_PATH = AI_DIR / "model" / "random_forest.pkl"


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("        AI FIRE DETECTION SYSTEM")
print("             PREDICTION MODULE")
print("=" * 60)

print("\nLoading trained Random Forest model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nERROR: Trained model not found.\n"
        f"Expected location:\n{MODEL_PATH}\n\n"
        "Please run train_rf.py first."
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully.")


# ============================================================
# 2. GET SENSOR INPUT
# ============================================================

print("\n" + "-" * 60)
print("ENTER SENSOR VALUES")
print("-" * 60)

try:

    temperature = float(
        input("\nTemperature (°C): ")
    )

    humidity = float(
        input("Humidity (%): ")
    )

    gas = float(
        input("Gas value: ")
    )

    flame = int(
        input("Flame (0 = No Flame, 1 = Flame): ")
    )

except ValueError:

    print(
        "\nERROR: Please enter valid numeric values."
    )

    exit()


# ============================================================
# 3. VALIDATE INPUT
# ============================================================

if flame not in [0, 1]:

    print(
        "\nERROR: Flame must be either 0 or 1."
    )

    exit()


# ============================================================
# 4. CREATE INPUT DATAFRAME
# ============================================================

sensor_data = pd.DataFrame({

    "temperature": [temperature],

    "humidity": [humidity],

    "gas": [gas],

    "flame": [flame]
})


# ============================================================
# 5. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    sensor_data
)

predicted_status = prediction[0]


# ============================================================
# 6. GET PREDICTION PROBABILITY
# ============================================================

probabilities = model.predict_proba(
    sensor_data
)

class_probabilities = dict(
    zip(
        model.classes_,
        probabilities[0]
    )
)


# ============================================================
# 7. DISPLAY SENSOR VALUES
# ============================================================

print("\n" + "=" * 60)
print("              SENSOR DATA")
print("=" * 60)

print(
    f"Temperature : {temperature:.2f} °C"
)

print(
    f"Humidity    : {humidity:.2f} %"
)

print(
    f"Gas Value   : {gas:.2f}"
)

print(
    f"Flame       : "
    f"{'DETECTED' if flame == 1 else 'NOT DETECTED'}"
)


# ============================================================
# 8. DISPLAY PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("              AI PREDICTION")
print("=" * 60)

print(
    f"\nPredicted Status: {predicted_status}"
)


# ============================================================
# 9. DISPLAY PROBABILITIES
# ============================================================

print("\nPrediction Probabilities:")

for class_name, probability in class_probabilities.items():

    print(
        f"  {class_name:10s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 10. FIRE ALERT
# ============================================================

print("\n" + "=" * 60)
print("                 RESULT")
print("=" * 60)

if predicted_status == "Critical":

    print("\n!!! CRITICAL FIRE ALERT !!!")
    print("Immediate action required.")

elif predicted_status == "Warning":

    print("\n!! FIRE WARNING !!")
    print("Please check the environment.")

else:

    print("\nSYSTEM STATUS: NORMAL")
    print("No immediate fire risk detected.")


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("Prediction completed successfully.")
print("=" * 60)