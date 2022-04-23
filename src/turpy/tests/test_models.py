from turpy.models import TfIdfClassifier, Doc2VecClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


def test_tfidf_classifier():
    X_input = pd.Series(["bu ayakkabıyı çok beğendim", "3 ay önce aldım beğenerek kullanıyorum", "ne iyi ne kötü", "ortalama bir saat",
                        "hayatımda bu kadar adi ve kötü plastik görmedim", "o kadar pahalı ki onun yerine başka markadan alırım"])
    y_input = pd.Series([1, 1, 0, 0, -1, -1])

    model = TfIdfClassifier()
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    assert np.array_equal(preds, y_input)

    model = TfIdfClassifier()
    model.prefit(X_input)
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    assert np.array_equal(preds, y_input)

    model = TfIdfClassifier(estimator=MLPClassifier())
    model.prefit(X_input)
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    probas = model.predict_proba(X_input)
    assert np.array_equal(preds, y_input)
    assert np.shape(probas) == (len(X_input), len(y_input.value_counts()))


def test_doc2vec_classifier():
    X_input = pd.Series(["bu ayakkabıyı çok beğendim", "3 ay önce aldım beğenerek kullanıyorum", "ne iyi ne kötü", "ortalama bir saat",
                        "hayatımda bu kadar adi ve kötü plastik görmedim", "o kadar pahalı ki onun yerine başka markadan alırım"])
    y_input = pd.Series([1, 1, 0, 0, -1, -1])

    model = Doc2VecClassifier()
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    assert np.array_equal(preds, y_input)

    model = Doc2VecClassifier()
    model.prefit(X_input)
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    assert np.array_equal(preds, y_input)

    model = Doc2VecClassifier(estimator=MLPClassifier())
    model.prefit(X_input)
    model.fit(X_input, y_input)
    preds = model.predict(X_input)
    probas = model.predict_proba(X_input)
    assert np.array_equal(preds, y_input)
    assert np.shape(probas) == (len(X_input), len(y_input.value_counts()))
