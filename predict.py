"""Classify new messages with the trained model.

Usage:
    python predict.py "Congratulations! You won a free prize, call now"
    python predict.py            (interactive mode)
"""

import os
import sys

import joblib

import preprocess  # noqa: F401  (needed so the saved pipeline can find clean_text)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "spam_model.joblib")


def classify(model, text):
    return "SPAM" if model.predict([text])[0] == 1 else "HAM (not spam)"


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit("Model not found. Run 'python train.py' first.")
    model = joblib.load(MODEL_PATH)

    if len(sys.argv) > 1:
        print(classify(model, " ".join(sys.argv[1:])))
        return
    print("Type a message to check (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        print(classify(model, text))


if __name__ == "__main__":
    main()
