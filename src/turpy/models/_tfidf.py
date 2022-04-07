from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

class TfIdfClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=RidgeClassifier):
        self.estimator = estimator
        self.vectorizer = TfidfVectorizer()
        self.is_prefit = False

    def prefit(self, X, y=None):
        if self.is_prefit:
            print("Overwriting previous prefit.")

        self.vectorizer.fit(X)
        self.is_prefit = True
        return self

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target (y) must be provided.")

        if self.is_prefit:
            X_vec = self.vectorizer.transform(X)
        else:
            X_vec = self.vectorizer.fit_transform(X)

        self.estimator.fit(X_vec, y)

        return self

    def predict(self, X, y=None):

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict(X_vec)

    def predict_proba(self, X, y=None):

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.estimator} does not have a predict_proba method.")

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict_proba(X_vec)
