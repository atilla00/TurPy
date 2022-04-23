import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator
from simpletransformers.classification import ClassificationModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from .._types import validate_text_input


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """Text Classifier using TF-IDF features given any Sklearn compatible estimator.

    Parameters
    ----------
    model_type : str, default="distilbert"
        The type of model (bert, xlnet, xlm, roberta, distilbert)
    model_name : str, default="dbmdz/distilbert-base-turkish-cased"
        The exact architecture and trained weights to use.
        This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
    use_gpu : bool, default=True
        Use GPU if available. Setting to False will force model to use CPU only.
    gpu_id : int, default=0
        Specific GPU that should be used.
    n_gpu : int, default=1
        Number of GPUs to use.
    silent : bool, default=False
        Show progress bar.
    batch_size : int, default=8
        Training batch_size.
    no_epochs : int, default=1
        The number of epochs the model will be trained for.
    learning_rate : float, default=4e-5
        The learning rate for training.
    max_seq_length : int, default=128
        Maximum sequence length the model will support.
    output_dir : str, default="/outputs"
        The directory where all outputs will be stored. This includes model checkpoints and evaluation results.
        Overwrites this directory if run again.
    """

    def __init__(self,
                 model_type: str = "distilbert",
                 model_path: str = "dbmdz/distilbert-base-turkish-cased",
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 n_gpu: int = 1,
                 silent: bool = False,
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
        self.silent = silent
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.output_dir = output_dir

        self.model_path = model_path
        self.encoder = LabelEncoder()

    def prefit(self, X: pd.Series, y=None):
        """Prefit Transformer model with unlabeled text data.

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

        raise NotImplementedError("Prefit method is not implemented for transformer models.")

    def fit(
        self,
        X: pd.Series,
        y: pd.Series,
        verbose: bool = True
    ):
        """Fit/Finetune a Transformer model from traning set.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : pd.Series
            Pandas text series containing targets.

        verbose : bool, default=True
            Training verbosity

        Returns
        -------
        self : object
            Fitted estimator.
        """

        validate_text_input(X)

        target_type = type_of_target(y)
        if target_type not in ["binary", "multiclass"]:
            raise ValueError(f"Type of target is not classification. Type: {target_type}")
        self.num_labels = y.nunique()

        # Simpletransformer dataset format
        train_df = pd.DataFrame()
        train_df["text"], train_df["labels"] = X, y

        # Map target to between 0 to n-1 classes
        train_df["labels"] = self.encoder.fit_transform(train_df["labels"])

        self.args = {
            "output_dir": self.output_dir,
            "overwrite_output_dir": True,

            "silent": self.silent,

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

    def predict(
        self,
        X: pd.Series,
        batch_size: int = 1,
        device: int = -1
    ):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        batch_size : int, default=1
            Size of batch.

        device : int, default=-1
            Device number for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions.
        """
        validate_text_input(X)

        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, batch_size=batch_size, device=device)

        preds = pipeline(X.tolist())  # List[Dict[]]
        preds[:] = [int(pred["label"].split("_")[-1]) for pred in preds]  # List[int]
        preds[:] = self.encoder.inverse_transform(preds)

        return preds

    def predict_proba(self, X: pd.Series, batch_size: int = 1, device: int = -1):
        """Probability estimates.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        batch_size : int, default=1
            Size of batch.

        device : int, default=-1
            Device number for prediction.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        validate_text_input(X)

        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True, batch_size=batch_size, device=device)
        predictions = pipeline(X.tolist())

        probs = []
        for prediction in predictions:
            scores = []
            for label in prediction:
                scores.append(label["score"])

            probs.append(scores)

        return np.ndarray(scores)
