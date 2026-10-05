from pathlib import Path
from ultralytics import YOLO


# ============================================================
# YOLO FIRE + SMOKE DETECTION TRAINING
# ============================================================

# train_yolo.py is inside ai/yolo/
YOLO_DIR = Path(__file__).resolve().parent

DATASET_DIR = YOLO_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# Pretrained YOLO model
MODEL_NAME = "yolov8n.pt"

# Training configuration
EPOCHS = 10
IMAGE_SIZE = 320
BATCH_SIZE = 8
WORKERS = 2


# ============================================================
# 1. CHECK DATASET
# ============================================================

def check_dataset():

    print("=" * 70)
    print("       YOLO FIRE + SMOKE DETECTION")
    print("=" * 70)

    print("\nChecking dataset...")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"ERROR: Dataset folder not found:\n{DATASET_DIR}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"ERROR: data.yaml not found:\n{DATA_YAML}"
        )

    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"

    test_images = DATASET_DIR / "test" / "images"
    test_labels = DATASET_DIR / "test" / "labels"

    required_folders = [
        train_images,
        train_labels,
        test_images,
        test_labels,
    ]

    for folder in required_folders:

        if not folder.exists():

            raise FileNotFoundError(
                f"ERROR: Required folder not found:\n{folder}"
            )

    train_image_count = len(
        list(train_images.glob("*"))
    )

    train_label_count = len(
        list(train_labels.glob("*.txt"))
    )

    test_image_count = len(
        list(test_images.glob("*"))
    )

    test_label_count = len(
        list(test_labels.glob("*.txt"))
    )

    print("\nDataset structure verified.")

    print(f"\nTraining images : {train_image_count}")
    print(f"Training labels : {train_label_count}")

    print(f"Test images     : {test_image_count}")
    print(f"Test labels     : {test_label_count}")

    if train_image_count == 0:
        raise ValueError("ERROR: No training images found.")

    if train_label_count == 0:
        raise ValueError("ERROR: No training labels found.")

    print("\nDataset check completed successfully.")


# ============================================================
# 2. TRAIN YOLO
# ============================================================

def train_model():

    print("\n" + "=" * 70)
    print("STARTING YOLO TRAINING")
    print("=" * 70)

    print(f"\nModel       : {MODEL_NAME}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Batch size  : {BATCH_SIZE}")

    # Load pretrained YOLO model
    model = YOLO(MODEL_NAME)

    # Train
    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        project=str(YOLO_DIR / "runs"),

        name="fire_smoke",

        exist_ok=True,

        patience=10,

        pretrained=True,

        verbose=True,
    )

    return model, results


# ============================================================
# 3. MAIN
# ============================================================

def main():

    check_dataset()

    model, results = train_model()

    print("\n" + "=" * 70)
    print("YOLO TRAINING COMPLETED")
    print("=" * 70)

    print("\nTraining results saved inside:")

    print(
        YOLO_DIR / "runs" / "fire_smoke"
    )

    print("\nBest trained model should be here:")

    print(
        YOLO_DIR
        / "runs"
        / "fire_smoke"
        / "weights"
        / "best.pt"
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()