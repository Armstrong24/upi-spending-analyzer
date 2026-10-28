"""AI part 1: transaction categoriser.

Model  : TF-IDF (character n-grams) + Logistic Regression  (scikit-learn)
Data   : ~2,500 labelled UPI descriptions (data/training_data.csv)
Fallback: keyword rules when the model's confidence is low.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from .merchants import KEYWORD_RULES, INCOME_CATEGORY

CONFIDENCE_THRESHOLD = 0.45


def clean_text(text: str) -> str:
    """Lower-case and remove UPI noise (reference numbers, bank codes, symbols)."""
    t = str(text).lower()
    t = re.sub(r"upi/(dr|cr)/\d+/", " ", t)
    t = re.sub(r"\d{6,}", " ", t)              # long reference numbers
    t = re.sub(r"/(yesb|hdfc|icic|sbin|utib|pytm|kkbk)/", " ", t)
    t = re.sub(r"[^a-z0-9@. ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                  ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])


@dataclass
class TrainedModel:
    pipeline: Pipeline
    accuracy: float
    report: dict
    confusion: np.ndarray
    labels: list = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    unseen_accuracy: float = None   # accuracy on merchants the model never saw


def train(training_df: pd.DataFrame, seed: int = 42) -> TrainedModel:
    X, y = training_df["description"], training_df["category"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    labels = sorted(y.unique())
    tm = TrainedModel(pipeline=model, accuracy=accuracy_score(y_te, pred),
                      report=classification_report(y_te, pred, output_dict=True, zero_division=0),
                      confusion=confusion_matrix(y_te, pred, labels=labels), labels=labels,
                      n_train=len(X_tr), n_test=len(X_te))
    # Harder, honest test: hide whole merchants from training
    if "merchant" in training_df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_i, te_i = next(gss.split(X, y, groups=training_df["merchant"]))
        m2 = build_model().fit(X.iloc[tr_i], y.iloc[tr_i])
        tm.unseen_accuracy = accuracy_score(y.iloc[te_i], m2.predict(X.iloc[te_i]))
    # refit on everything for real use
    tm.pipeline = build_model().fit(X, y)
    return tm


def rule_category(description: str):
    t = clean_text(description)
    for category, words in KEYWORD_RULES.items():
        if any(w in t for w in words):
            return category
    # "Paid to Firstname Lastname" or a personal-looking VPA -> transfer to a person
    if re.match(r"^(paid to|to|sent to|payment to) [a-z]+( [a-z]+)?$", t) or \
       re.search(r"@ok(hdfcbank|sbi|axis|icici)$", t):
        return "Transfers to People"
    return None


def categorize(df: pd.DataFrame, model: TrainedModel) -> pd.DataFrame:
    """Add Category, Confidence and Method columns."""
    out = df.copy()
    debit = out["type"] == "DEBIT"
    out["category"], out["confidence"], out["method"] = INCOME_CATEGORY, 1.0, "rule (credit)"
    if debit.any():
        descs = out.loc[debit, "description"]
        proba = model.pipeline.predict_proba(descs)
        classes = model.pipeline.classes_
        best = proba.argmax(axis=1)
        cats, confs, methods = [], [], []
        for d, i, p in zip(descs, best, proba):
            conf = float(p[i])
            if conf >= CONFIDENCE_THRESHOLD:
                cats.append(classes[i]); methods.append("ML model")
            else:
                r = rule_category(d)
                cats.append(r or classes[i]); methods.append("keyword rule" if r else "ML (low confidence)")
            confs.append(round(conf, 2))
        out.loc[debit, "category"] = cats
        out.loc[debit, "confidence"] = confs
        out.loc[debit, "method"] = methods
    return out


def predict_one(text: str, model: TrainedModel):
    p = model.pipeline.predict_proba([text])[0]
    order = p.argsort()[::-1][:3]
    return [(model.pipeline.classes_[i], float(p[i])) for i in order]
