import argparse
from pathlib import Path

from qa_grader.preprocess import build_dataset
from qa_grader.train import train_model
from qa_grader.evaluate import evaluate_model
from qa_grader.predict import grade_file, grade_transcript
from qa_grader.config import DATASET_FILE, MODEL_FILE
from qa_grader.api import app
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="QA Call Grader CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-dataset
    subparsers.add_parser("build-dataset", help="Build dataset from data/raw/ transcripts")

    # train
    subparsers.add_parser("train", help="Train model on dataset")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate trained model")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Grade a transcript file")
    predict_parser.add_argument("file", type=str, help="Path to transcript .txt file")

    # api
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument("--host", type=str, default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "build-dataset":
        build_dataset()

    elif args.command == "train":
        if not DATASET_FILE.exists():
            print("[ERROR] No dataset found. Run 'cli.py build-dataset' first.")
            return
        train_model()

    elif args.command == "evaluate":
        if not MODEL_FILE.exists():
            print("[ERROR] No trained model found. Run 'cli.py train' first.")
            return
        evaluate_model()

    elif args.command == "predict":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return
        grade = grade_file(file_path)
        print(f"[RESULT] {file_path.name} → {grade.upper()}")

    elif args.command == "api":
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
