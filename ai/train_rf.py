from pathlib import Path
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


# ============================================================
# RANDOM FOREST TRAINING - AI FIRE DETECTION SYSTEM
# ============================================================

# train_rf.py is inside the ai/ folder
AI_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

DATASET_PATH = AI_DIR / "dataset" / "fire_data.csv"

MODEL_DIR = AI_DIR / "model"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
INFO_PATH = MODEL_DIR / "model_info.pkl"
RESULTS_PATH = MODEL_DIR / "training_results.txt"
CM_PATH = MODEL_DIR / "confusion_matrix.csv"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

FEATURES = [
    "temperature",
    "humidity",
    "gas",
    "flame"
]

# Target column in fire_data.csv
TARGET = "label"

EXPECTED_LABELS = [
    "Normal",
    "Warning",
    "Critical"
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200


# ============================================================
# 1. LOAD AND VALIDATE DATASET
# ============================================================

def load_and_validate_dataset():

    print("=" * 70)
    print("AI FIRE DETECTION SYSTEM - RANDOM FOREST TRAINING")
    print("=" * 70)

    print("\nDataset path:")
    print(DATASET_PATH)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "\nERROR: fire_data.csv not found.\n\n"
            f"Expected location:\n{DATASET_PATH}\n\n"
            "Required structure:\n"
            "AI-Fire-Detection-System/\n"
            "|-- ai/\n"
            "|   |-- dataset/\n"
            "|   |   `-- fire_data.csv\n"
            "|   |-- model/\n"
            "|   `-- train_rf.py\n"
            "`-- README.md"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    if df.empty:

        raise ValueError(
            "ERROR: fire_data.csv is empty."
        )

    print("\nDataset loaded successfully.")

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nERROR: Missing columns: {missing_columns}\n"
            f"Required columns: {required_columns}"
        )

    # Keep only required columns
    df = df[required_columns].copy()

    # --------------------------------------------------------
    # Convert sensor values to numbers
    # --------------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # --------------------------------------------------------
    # Check missing sensor values
    # --------------------------------------------------------

    missing_values = df[FEATURES].isna().sum()

    if missing_values.any():

        bad_columns = missing_values[
            missing_values > 0
        ]

        raise ValueError(
            "\nERROR: Missing or non-numeric sensor values found:\n"
            f"{bad_columns}"
        )

    # --------------------------------------------------------
    # Check valid labels
    # --------------------------------------------------------

    invalid_labels = sorted(
        set(df[TARGET]) - set(EXPECTED_LABELS)
    )

    if invalid_labels:

        raise ValueError(
            f"\nERROR: Invalid labels found: {invalid_labels}\n"
            f"Allowed labels: {EXPECTED_LABELS}"
        )

    # --------------------------------------------------------
    # Check all classes
    # --------------------------------------------------------

    missing_labels = sorted(
        set(EXPECTED_LABELS)
        - set(df[TARGET].unique())
    )

    if missing_labels:

        raise ValueError(
            f"\nERROR: These classes are missing: "
            f"{missing_labels}"
        )

    # --------------------------------------------------------
    # Check flame values
    # --------------------------------------------------------

    flame_values = set(
        df["flame"].unique()
    )

    if not flame_values.issubset({0, 1}):

        raise ValueError(
            "\nERROR: flame must contain only 0 or 1.\n"
            f"Found values: {flame_values}"
        )

    # --------------------------------------------------------
    # Minimum dataset check
    # --------------------------------------------------------

    if len(df) < 100:

        raise ValueError(
            f"\nERROR: Only {len(df)} rows found.\n"
            "At least 100 rows are required."
        )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print(f"\nTotal rows: {len(df)}")

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nClass distribution:")

    print(
        df[TARGET]
        .value_counts()
        .reindex(
            EXPECTED_LABELS,
            fill_value=0
        )
    )

    print("\nDataset validation successful.")

    return df


# ============================================================
# 2. TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(df):

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df[FEATURES]
    y = df[TARGET]

    print("\n" + "-" * 70)
    print("TRAIN / TEST SPLIT")
    print("-" * 70)

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINING RANDOM FOREST")
    print("-" * 70)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed successfully.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=EXPECTED_LABELS,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=EXPECTED_LABELS
    )

    # --------------------------------------------------------
    # 5-Fold Cross Validation
    # --------------------------------------------------------

    print("\nRunning 5-fold cross-validation...")

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        pd.Series(
            model.feature_importances_,
            index=FEATURES
        )
        .sort_values(
            ascending=False
        )
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("MODEL EVALUATION")
    print("-" * 70)

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\n5-Fold Cross Validation:"
    )

    print(
        f"Mean Accuracy: "
        f"{cv_scores.mean() * 100:.2f}%"
    )

    print(
        f"Standard Deviation: "
        f"{cv_scores.std() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{label}"
            for label in EXPECTED_LABELS
        ],
        columns=[
            f"Predicted_{label}"
            for label in EXPECTED_LABELS
        ]
    )

    print(confusion_df)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nFeature Importance:")

    for feature, value in importance.items():

        print(
            f"  {feature:12s}: "
            f"{value:.4f}"
        )

    return (
        model,
        accuracy,
        report,
        cm,
        cv_scores,
        importance
    )


# ============================================================
# 3. SAVE MODEL AND RESULTS
# ============================================================

def save_outputs(
    model,
    accuracy,
    report,
    cm,
    cv_scores,
    importance,
    row_count
):

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    # --------------------------------------------------------
    # Save model information
    # --------------------------------------------------------

    model_info = {

        "features": FEATURES,

        "target": TARGET,

        "classes": list(
            model.classes_
        ),

        "test_size": TEST_SIZE,

        "random_state": RANDOM_STATE,

        "n_estimators": N_ESTIMATORS
    }

    with open(
        INFO_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_info,
            file
        )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{label}"
            for label in EXPECTED_LABELS
        ],
        columns=[
            f"Predicted_{label}"
            for label in EXPECTED_LABELS
        ]
    )

    confusion_df.to_csv(
        CM_PATH
    )

    # --------------------------------------------------------
    # Save training results
    # --------------------------------------------------------

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI FIRE DETECTION SYSTEM\n"
        )

        file.write(
            "RANDOM FOREST TRAINING RESULTS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Dataset: {DATASET_PATH}\n"
        )

        file.write(
            f"Rows used: {row_count}\n"
        )

        file.write(
            f"Features: {FEATURES}\n"
        )

        file.write(
            f"Target: {TARGET}\n"
        )

        file.write(
            f"Classes: {EXPECTED_LABELS}\n"
        )

        file.write(
            f"Test accuracy: "
            f"{accuracy * 100:.2f}%\n"
        )

        file.write(
            f"5-fold CV accuracy: "
            f"{cv_scores.mean() * 100:.2f}% "
            f"+/- "
            f"{cv_scores.std() * 100:.2f}%\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            report
        )

        file.write(
            "\nFeature Importance\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for feature, value in importance.items():

            file.write(
                f"{feature}: "
                f"{value:.6f}\n"
            )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nModel saved   : {MODEL_PATH}"
    )

    print(
        f"Metadata saved: {INFO_PATH}"
    )

    print(
        f"Results saved : {RESULTS_PATH}"
    )

    print(
        f"Confusion     : {CM_PATH}"
    )


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

def main():

    # Load and validate dataset
    df = load_and_validate_dataset()

    # Train Random Forest
    (
        model,
        accuracy,
        report,
        cm,
        cv_scores,
        importance
    ) = train_random_forest(df)

    # Save model and results
    save_outputs(
        model,
        accuracy,
        report,
        cm,
        cv_scores,
        importance,
        len(df)
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()