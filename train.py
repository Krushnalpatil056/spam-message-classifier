"""Train and compare spam classifiers.

Usage:
    python train.py
    python train.py --data data/sms.tsv --label-col label --text-col message
Any file with a label column (ham/spam) and a text column works, e.g. an email dataset.
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")                      # save plots without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(path, label_col, text_col):
    separator = "\t" if path.endswith((".tsv", ".txt")) else ","
    if path.endswith((".tsv", ".txt")) and label_col == "label":
        # the bundled sms.tsv has no header row
        df = pd.read_csv(path, sep=separator, names=["label", "message"], header=None)
    else:
        df = pd.read_csv(path, sep=separator)
    df = df[[label_col, text_col]].dropna().drop_duplicates()
    df["target"] = (df[label_col].str.lower() == "spam").astype(int)   # spam = 1, ham = 0
    return df[text_col], df["target"]


def build_models():
    """Each model is a Pipeline: TF-IDF vectorizer -> classifier."""
    def tfidf():
        return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=1000, C=10))]),
        "SVM (LinearSVC)": Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(C=1.0))]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "data", "sms.tsv"))
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--text-col", default="message")
    args = parser.parse_args()

    X, y = load_data(args.data, args.label_col, args.text_col)
    print(f"Loaded {len(X)} messages | spam share: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results, fitted = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = (model, pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })

    table = pd.DataFrame(results).set_index("Model").round(4)
    print("\nModel comparison (test set):\n", table)

    best_name = table["F1"].idxmax()
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model by F1-score: {best_name}")
    print(classification_report(y_test, best_pred, target_names=["ham", "spam"]))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, best_pred))

    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
    joblib.dump(best_model, os.path.join(BASE_DIR, "model", "spam_model.joblib"))
    table.to_csv(os.path.join(BASE_DIR, "model", "results.csv"))

    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=["ham", "spam"],
                                            cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(os.path.join(BASE_DIR, "model", "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    print("\nSaved model, results.csv and confusion_matrix.png in the 'model' folder.")


if __name__ == "__main__":
    main()
