import gensim
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator


def _gensim_preprocess(X):
    """Lowers and splits texts"""
    X_pre = X.apply(gensim.utils.simple_preprocess).tolist()
    X_pre = [gensim.models.doc2vec.TaggedDocument(texts, [i]) for i, texts in enumerate(X_pre)]

    return X_pre


class Doc2VecClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, vector_size, min_count, epochs, estimator=RidgeClassifier()):
        self.estimator = estimator
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs

        self.vectorizer = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)
        self.is_prefit = False

    def prefit(self, X):
        if self.is_prefit:
            print("Overwriting previous prefit.")

        train_corpus = _gensim_preprocess(X)

        self.vectorizer.build_vocab(train_corpus)
        self.vectorizer.train(train_corpus, total_examples=self.vectorizer.corpus_count, epochs=self.vectorizer.epochs)

        self.is_prefit = True

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Target(y) must be provided.")

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

    def predict(self, X, y=None):
        X_pre = _gensim_preprocess(X)
        vectors = [self.vectorizer.infer_vector(doc.words) for doc in X_pre]
        X_vec = pd.DataFrame(np.stack(vectors))

        return self.estimator.predict(X_vec)

    def predict_proba(self, X, y=None):

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.estimator} does not have a predict_proba method.")

        X_pre = _gensim_preprocess(X)
        vectors = [self.vectorizer.infer_vector(doc.words) for doc in X_pre]
        X_vec = pd.DataFrame(np.stack(vectors))

        return self.estimator.predict_proba(X_vec)
