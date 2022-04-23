import gensim
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from .._types import validate_text_input
from ..base import AnySklearnEstimator


def _gensim_preprocess(X: pd.Series):
    """Lowers and splits texts"""
    X_pre = X.apply(gensim.utils.simple_preprocess).tolist()
    X_pre = [gensim.models.doc2vec.TaggedDocument(texts, [i]) for i, texts in enumerate(X_pre)]

    return X_pre


class Doc2VecClassifier(BaseEstimator, ClassifierMixin):
    """Text Classifier using Doc2Vec feateures given any Sklearn compatible estimator.

    Parameters
    ----------
    estimator : AnySklearnEstimator, default=RidgeClassifier
        Estimator to fit on TF-IDF vector. Prefer estimators with sparse vector support.

    vector_size : int, default=100
        Dimensionality of the feature vectors.

    min_count : int, default=1
        Ignores all words with total frequency lower than this.

    window : int, default=5
        The maximum distance between the current and predicted word within a sentence.

    workers : int, default=1
        Use these many worker threads to train the model (=faster training with multicore machines).

    epochs : int, default=10
        Number of iterations (epochs) over the corpus.
    """

    def __init__(self,
                 estimator: AnySklearnEstimator = RidgeClassifier(),
                 vector_size: int = 100,
                 min_count: int = 1,
                 window: int = 5,
                 epochs: int = 10,
                 workers: int = 1,
                 ):

        self.estimator = estimator
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.workers = workers

        self.vectorizer = gensim.models.doc2vec.Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            window=self.window,
            epochs=self.epochs,
            workers=self.workers,
        )
        self.is_prefit = False

    def prefit(
        self,
        X: pd.Series,
        y=None
    ):
        """Prefit Doc2Vec vectorizer with unlabeled text data.

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

        train_corpus = _gensim_preprocess(X)

        self.vectorizer.build_vocab(train_corpus)
        self.vectorizer.train(train_corpus, total_examples=self.vectorizer.corpus_count, epochs=self.vectorizer.epochs)

        self.is_prefit = True

    def fit(
        self,
        X: pd.Series,
        y: pd.Series
    ):
        """Build Doc2Vec vectors from training set and fit the provided estimator.

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

        if self.is_prefit:
            X_pre = _gensim_preprocess(X)
            vectors = [self.vectorizer.infer_vector(doc.words) for doc in X_pre]
            X_vec = pd.DataFrame(np.stack(vectors))
        else:
            train_corpus = _gensim_preprocess(X)
            self.vectorizer.build_vocab(train_corpus)
            self.vectorizer.train(train_corpus, total_examples=self.vectorizer.corpus_count, epochs=self.vectorizer.epochs)

            vectors = [self.vectorizer.infer_vector(doc.words) for doc in train_corpus]
            X_vec = pd.DataFrame(np.stack(vectors))

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

        X_pre = _gensim_preprocess(X)
        vectors = [self.vectorizer.infer_vector(doc.words) for doc in X_pre]
        X_vec = pd.DataFrame(np.stack(vectors))

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
            raise AttributeError(f"{self.estimator} does not have a predict_proba method.")

        X_pre = _gensim_preprocess(X)
        vectors = [self.vectorizer.infer_vector(doc.words) for doc in X_pre]
        X_vec = pd.DataFrame(np.stack(vectors))

        return self.estimator.predict_proba(X_vec)
