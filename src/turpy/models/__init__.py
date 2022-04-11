from ._tfidf import TfIdfClassifier
from ._doc2vec import Doc2VecClassifier

try:
    from ._deep import TransformerClassifier
except ModuleNotFoundError:
    pass
