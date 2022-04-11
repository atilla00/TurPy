try:
    import torch
except ModuleNotFoundError:
    raise ImportError("Please install Pytorch beforehand.")

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator
from simpletransformers.classification import ClassificationModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 model_type: str = "distilbert",
                 model_path: str = "dbmdz/distilbert-base-turkish-cased",
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 n_gpu: int = 1,
                 batch_size: int = 8,
                 no_epochs: int = 1,
                 learning_rate: float = 4e-5,
                 max_seq_length: int = 128,
                 output_dir: str = "/outputs"
                 ):

        self.model_type = model_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.n_gpu = n_gpu
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.output_dir = output_dir

        self.model_path = model_path
        self.encoder = LabelEncoder()

    def fit(self, X, y, verbose=True):

        # Simpletransformer dataset format
        train_df = pd.DataFrame()
        train_df["text"], train_df["labels"] = X, y

        # Map target to between 0 to n-1 classes
        train_df["labels"] = self.encoder.fit_transform(train_df["labels"])

        target_type = type_of_target(y)
        if not target_type in ["binary", "multiclass"]:
            raise ValueError(f"Type of target is not classification. Type: {target_type}")
        self.num_labels = y.nunique()

        self.args = {
            "output_dir": self.output_dir,
            "overwrite_output_dir": True,

            "max_seq_length": self.max_seq_length,
            "num_train_epochs": self.no_epochs,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.batch_size,
            "eval_batch_size": self.batch_size,
        }

        model = ClassificationModel(self.model_type, self.model_path, use_cuda=self.use_gpu,
                                    cuda_device=self.gpu_id, num_labels=self.num_labels, args=self.args)

        model.train_model(train_df, verbose=verbose)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.args["output_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.args["output_dir"])

        return self

    def predict(self, X, batch_size=1, device=-1):
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, batch_size=batch_size, device=device)
        preds = pipeline(X.tolist())

        preds = [int(pred["label"].split("_")[-1]) for pred in preds]
        preds = self.encoder.inverse_transform(preds)

        return preds

    def predict_proba(self, X, batch_size=1, device=-1):
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True, batch_size=batch_size, device=device)

        predictions = pipeline(X.tolist())

        probs = []
        for prediction in predictions:
            scores = []
            for label in prediction:
                scores.append(label["score"])

            probs.append(scores)

        return np.array(probs)
