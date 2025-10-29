from __future__ import annotations
import os, joblib
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = os.getenv("MODEL_PATH", "models/spam_nb.joblib")

class SpamModel:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    def load_or_bootstrap(self) -> None:
        """
        Loads the A2 model if available; otherwise trains a tiny fallback so the app
        always runs for the demo. Replace fallback with your real A2 artefact.
        """
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            return

        # --- Fallback demo model (tiny, replace with real artefact) ---
        X = [
            "win money now", "click here claim prize", "urgent free offer",
            "limited time deal", "you have been selected", "buy now discount coupon",
            "call me when you arrive", "let's meet tomorrow", "see you at lunch",
            "please review the document", "the report is attached", "thanks for your help",
        ]
        y = ["spam","spam","spam","spam","spam","spam","ham","ham","ham","ham","ham","ham"]

        vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
        clf = MultinomialNB()
        self.pipeline = Pipeline([("tfidf", vec), ("nb", clf)])
        self.pipeline.fit(X, y)

    def predict(self, text: str) -> tuple[str, float, list[tuple[str, float]], int]:
        assert self.pipeline is not None, "Model not loaded"
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        label = classes[proba.argmax()]
        prob = float(proba.max())

        # simple token importance proxy (NOT full SHAP: good enough for charts)
        vec = self.pipeline.named_steps["tfidf"]
        nb  = self.pipeline.named_steps["nb"]
        Xv  = vec.transform([text])
        # log prob for each feature for predicted class:
        cls_index = list(classes).index(label)
        log_probs = nb.feature_log_prob_[cls_index]  # shape [n_features]
        nz_idx = Xv.nonzero()[1]
        scores = []
        for j in nz_idx:
            tok = vec.get_feature_names_out()[j]
            scores.append((tok, float(log_probs[j])))
        scores.sort(key=lambda t: t[1], reverse=True)
        top = scores[:15]
        return label, prob, top, int(vec.max_features_ or Xv.shape[1])

MODEL = SpamModel()
