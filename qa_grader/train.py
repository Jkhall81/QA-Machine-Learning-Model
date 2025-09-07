import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from qa_grader.config import DATASET_FILE, MODEL_FILE

def train_model():
    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    X = df["transcript_text"]
    y = df["label"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    # Train
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"[INFO] Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
