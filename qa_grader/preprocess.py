import os
import glob
import pandas as pd
from pathlib import Path
from qa_grader.config import RAW_DATA_DIR, DATASET_FILE, LABELS

def clean_text(text: str) -> str:
    """
    Basic transcript cleaning:
    - Lowercase
    - Strip extra whitespace
    - Remove leading/trailing noise
    """
    text = text.strip().replace("\n", " ")
    text = " ".join(text.split())  # collapse multiple spaces
    return text


def build_dataset() -> pd.DataFrame:
    """
    Build a labeled dataset from raw .txt files.
    Folder structure must be:
        data/raw/pass/*.txt
        data/raw/fail/*.txt
        data/raw/warning/*.txt
    Returns a DataFrame with transcript_text + label.
    """
    rows = []

    for label in LABELS:
        folder = RAW_DATA_DIR / label
        if not folder.exists():
            print(f"[WARN] Skipping missing folder: {folder}")
            continue

        for file in glob.glob(str(folder / "*.txt")):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                rows.append({
                    "transcript_text": clean_text(text),
                    "label": label
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No transcripts found. Check your data/raw/ structure.")

    # Save dataset
    os.makedirs(DATASET_FILE.parent, exist_ok=True)
    df.to_csv(DATASET_FILE, index=False)

    print(f"[INFO] Dataset built with {len(df)} transcripts → {DATASET_FILE}")
    return df


if __name__ == "__main__":
    build_dataset()
