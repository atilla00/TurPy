import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from .._types import check_input
from typing import Tuple, Any, Union

AnySklearnEstimator = Any

class TfIdfClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 estimator: AnySklearnEstimator = RidgeClassifier(),
                 ngram_range: Tuple[int, int] = (1, 1),
                 max_df: Union[float, int] = 1,
                 min_df: Union[float, int] = 1,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False,
                 lowercase: bool = False
                 ):

        self.is_prefit = False
        self.estimator = estimator
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, max_df=max_df, min_df=min_df, lowercase=lowercase,
            use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )

    def prefit(self, X: pd.Series):
        check_input(X)

        if self.is_prefit:
            print("Overwriting previous prefit.")

        self.vectorizer.fit(X)
        self.is_prefit = True
        return self

    def fit(self, X: pd.Series, y: pd.Series):
        check_input(X)

        if y is None:
            raise ValueError("Target (y) must be provided.")

        if self.is_prefit:
            X_vec = self.vectorizer.transform(X)
        else:
            X_vec = self.vectorizer.fit_transform(X)

        self.estimator.fit(X_vec, y)

        return self

    def predict(self, X: pd.Series):
        check_input(X)

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict(X_vec)

    def predict_proba(self, X: pd.Series):
        check_input(X)

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.estimator} does not have a predict_proba method.")

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict_proba(X_vec)
