import os
from pathlib import Path

# Base directory (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Labels
LABELS = ["pass", "warning", "fail"]

# Rule-based constants
REQUIRED_PHRASES = {
    "recorded_line": "on a recorded line",   # must be present
    "email": "email"                         # must be present (Q20)
}

# Output dataset file
DATASET_FILE = PROCESSED_DATA_DIR / "qa_transcripts.csv"

# Saved model file
MODEL_FILE = MODEL_DIR / "qa_model.pkl"
