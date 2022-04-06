from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

class TfIdfClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=RidgeClassifier):
        self.estimator = estimator
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target (y) must be provided.")

        X_vec = self.vectorizer.fit_transform(X)

        self.estimator.fit(X_vec, y)

        return self

    def transform(self, X, y=None):

        return self.estimator.transform(X)

    def predict(self, X, y=None):

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict(X_vec)

    def predict_proba(self, X, y=None):

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.estimator} does not have a predict_proba method.")

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict_proba(X_vec)
