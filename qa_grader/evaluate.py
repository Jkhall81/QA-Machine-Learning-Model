import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from qa_grader.config import DATASET_FILE, MODEL_FILE

def evaluate_model():
    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    X = df["transcript_text"]
    y = df["label"]

    # Split train/test (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Print report
    print("[INFO] Evaluation results:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
