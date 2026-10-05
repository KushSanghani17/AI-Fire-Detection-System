"""
Root-level wrapper to ensure 'import predict' always succeeds
regardless of working directory or how app.py was launched.
"""
from pathlib import Path
import sys

_AI_DIR = Path(__file__).resolve().parent / "ai"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from ai.predict import (
    load_model,
    get_thingspeak_data,
    predict_fire_status,
    send_prediction_to_thingspeak,
    display_result,
    main,
    MODEL_PATH,
    THINGSPEAK_CHANNEL_ID,
    THINGSPEAK_READ_URL,
    THINGSPEAK_WRITE_URL
)

__all__ = [
    "load_model",
    "get_thingspeak_data",
    "predict_fire_status",
    "send_prediction_to_thingspeak",
    "display_result",
    "main",
    "MODEL_PATH",
    "THINGSPEAK_CHANNEL_ID",
    "THINGSPEAK_READ_URL",
    "THINGSPEAK_WRITE_URL"
]
