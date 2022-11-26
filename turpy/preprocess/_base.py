import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BaseTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, to_replace: str = ""):
        self.to_replace = to_replace
        
    def fit(self, X: pd.Series, y: pd.Series = None):
        return self

    def transform(self, X: pd.Series, y: pd.Series = None):
        return X.str.replace(self.pattern, self.to_replace, regex=True)