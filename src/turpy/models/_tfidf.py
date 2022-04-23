import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Tuple, Union

from .._types import validate_text_input
from ..base import AnySklearnEstimator


class TfIdfClassifier(BaseEstimator, ClassifierMixin):
    """Text Classifier using TF-IDF features given any Sklearn compatible estimator.

    Parameters
    ----------
    estimator : AnySklearnEstimator, default=RidgeClassifier
        Estimator to fit on TF-IDF vector. Prefer estimators with sparse vector support.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.

    max_df : float or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float in range of [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts.

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    lowercase : bool, default=False
        Convert all characters to lowercase before tokenizing.
    """

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

    def prefit(
        self,
        X: pd.Series,
        y=None
    ):
        """Prefit Tf-Idf vectorizer with unlabeled text data.

        Parameters
        ----------
        X : pd.Series
            Pandas series containing texts.

        y : None
             This parameter is not needed.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        validate_text_input(X)

        if self.is_prefit:
            print("Overwriting previous prefit.")

        self.vectorizer.fit(X)
        self.is_prefit = True

        return self

    def fit(
        self,
        X: pd.Series,
        y: pd.Series
    ):
        """Build TF-IDF vectors from training set and fit the provided estimator. If prefit, it uses prefitted vectorizer to embed input text.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : pd.Series
            Pandas text series containing targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        validate_text_input(X)

        if y is None:
            raise ValueError("Target (y) must be provided.")

        if self.is_prefit:
            X_vec = self.vectorizer.transform(X)
        else:
            X_vec = self.vectorizer.fit_transform(X)

        self.estimator.fit(X_vec, y)

        return self

    def predict(
        self,
        X: pd.Series
    ):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions.
        """
        validate_text_input(X)

        X_vec = self.vectorizer.transform(X)
        return self.estimator.predict(X_vec)

    def predict_proba(
        self,
        X: pd.Series
    ):
        """Probability estimates.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """

        validate_text_input(X)

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Underlying estimator `{self.estimator}` does not support predict_proba")

        X_vec = self.vectorizer.transform(X)

        return self.estimator.predict_proba(X_vec)
