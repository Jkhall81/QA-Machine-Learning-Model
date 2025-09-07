import sys
import joblib
import pandas as pd
from pathlib import Path
from qa_grader.config import MODEL_FILE
from qa_grader.rules import check_rules
from qa_grader.preprocess import clean_text

def grade_transcript(text: str) -> str:
    """
    Grade a transcript using rules + ML model.
    Returns one of: "pass", "fail", "warning".
    """
    # 1. Rule-based check
    rule_result = check_rules(text)
    if rule_result:  # "fail" or "warning"
        return rule_result

    # 2. ML model
    model = joblib.load(MODEL_FILE)
    cleaned = clean_text(text)
    grade = model.predict([cleaned])[0]
    return grade


def grade_file(file_path: Path) -> str:
    """
    Load a transcript from file and grade it.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return grade_transcript(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m qa_grader.predict <transcript_file.txt>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    grade = grade_file(file_path)
    print(f"[RESULT] {file_path.name} → {grade.upper()}")
