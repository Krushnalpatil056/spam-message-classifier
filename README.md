# Spam Classifier (TF-IDF + Scikit-learn)

Classifies text messages as spam or ham (not spam) and compares three models.

## Run
```bash
pip install -r requirements.txt
python train.py                     # trains, compares models, saves the best one
python predict.py "You won a free prize! Call now"
```

## Dataset
`data/sms.tsv` is the public SMS Spam Collection (5,574 labelled messages; UCI Machine Learning Repository).
It is a message dataset, so the same pipeline works for emails: pass any CSV with a label column (`ham`/`spam`) and a text column:
```bash
python train.py --data data/emails.csv --label-col label --text-col text
```

## What it does
1. Cleans text with regular expressions (lowercase, URLs, numbers, punctuation, stop words) – `preprocess.py`
2. Builds a Scikit-learn `Pipeline`: `TfidfVectorizer` -> classifier
3. Trains Naive Bayes, Logistic Regression and Linear SVM on an 80/20 stratified split
4. Reports accuracy, precision, recall, F1 and a confusion matrix

## Results from the included run (test set, 1,034 messages)
| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Naive Bayes | 0.9874 | 0.9758 | 0.9237 | 0.9490 |
| Logistic Regression | 0.9836 | 0.9831 | 0.8855 | 0.9317 |
| SVM (LinearSVC) | 0.9865 | 0.9835 | 0.9084 | 0.9444 |

Your numbers may differ slightly on a different Scikit-learn version.
