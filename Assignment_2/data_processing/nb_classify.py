# nb_classify.py — Multinomial Naive Bayes on TF-IDF
# Uses: outputs/processed/emails_merged.train.csv / emails_merged.test.csv

import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "outputs" / "processed" / "emails_merged.train.csv"
TEST  = ROOT / "outputs" / "processed" / "emails_merged.test.csv"

def load_xy(path):
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), df["label"].astype(int).values

X_train, y_train = load_xy(TRAIN)
X_test,  y_test  = load_xy(TEST)

# TF-IDF keeps digits; no stopword removal (your analysis showed some function words are informative)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1,2))),   # unigrams+bigrams
    ("nb", MultinomialNB(alpha=0.5))                           # light smoothing
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print("=== Naive Bayes (TF-IDF) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}\n")
print("Confusion matrix [ [TN FP] [FN TP] ]")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed report:")
print(classification_report(y_test, y_pred, digits=4))