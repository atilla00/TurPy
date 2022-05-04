# mypy: ignore-errors
import math
import numpy as np
import pandas as pd
import torch
import os
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator
from simpletransformers.classification import ClassificationModel
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from .._types import validate_text_input

import logging
logger = logging.getLogger(__name__)


class TurpyClassificationModel(ClassificationModel):
    def __init__(self, model_type, model_name, tokenizer_type=None, tokenizer_name=None, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, onnx_execution_provider=None, **kwargs):
        super().__init__(model_type, model_name, tokenizer_type, tokenizer_name, num_labels, weight, args, use_cuda, cuda_device, onnx_execution_provider, **kwargs)

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if self.args.no_save:
            return
        return super().save_model(output_dir, optimizer, scheduler, model, results)

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False):
        if self.args.no_save:
            no_cache = True
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

    def train(
        self,
        train_dataloader,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        test_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        loss, *_ = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                else:
                    loss, *_ = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_df,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        #training_progress_scores = cast(training_progress_scores, Dict[List[int]])

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])

                        if test_df is not None:
                            test_results, _, _ = self.eval_model(
                                test_df,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )
                            for key in test_results:
                                training_progress_scores["test_" + key].append(
                                    test_results[key]
                                )

                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                best_eval_metric - results[args.early_stopping_metric]
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(
                    eval_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                if test_df is not None:
                    test_results, _, _ = self.eval_model(
                        test_df,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )
                    for key in test_results:
                        training_progress_scores["test_" + key].append(
                            test_results[key]
                        )

                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        best_eval_metric - results[args.early_stopping_metric]
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

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
    gradient_accumulation_steps : int, default=1
        The number of training steps to execute before performing a optimizer.step().
        Sacrificing training time to lower memory consumption.
    no_epochs : int, default=1
        The number of epochs the model will be trained for.
    learning_rate : float, default=4e-5
        The learning rate for training.
    max_seq_length : int, default=128
        Maximum sequence length the model will support.
    no_save : bool, default=False
        To save the model or not. If this is true, no files will be saved on disk and output_dir/cache_dir will be ignored.
    output_dir : str, default="/outputs"
        The directory where all outputs will be stored. This includes model checkpoints and evaluation results.
        Overwrites this directory if run again.
    cache_dir : str, default="/cache_dir"
        The directory where cached files will be saved.
    """

    def __init__(self,
                 model_type: str = "distilbert",
                 model_path: str = "dbmdz/distilbert-base-turkish-cased",
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 n_gpu: int = 1,
                 silent: bool = False,
                 batch_size: int = 8,
                 gradient_accumulation_steps: int = 1,
                 no_epochs: int = 1,
                 learning_rate: float = 4e-5,
                 max_seq_length: int = 128,
                 no_save: bool = True,
                 output_dir: str = "/outputs",
                 cache_dir: str = "/cache_dir"
                 ):

        self.model_type = model_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.n_gpu = n_gpu
        self.silent = silent
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.no_epochs = no_epochs
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.no_save = no_save
        self.output_dir = output_dir
        self.cache_dir = cache_dir

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
        train_df = pd.concat([X, y], ignore_index=True, axis=1)
        train_df.columns = ["text", "labels"]

        # Map target to between 0 to n-1 classes
        train_df["labels"] = self.encoder.fit_transform(train_df["labels"])

        self.args = {
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "overwrite_output_dir": True,

            "silent": self.silent,

            "max_seq_length": self.max_seq_length,
            "num_train_epochs": self.no_epochs,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.batch_size,
            "eval_batch_size": self.batch_size,
        }

        model = TurpyClassificationModel(self.model_type, self.model_path, use_cuda=self.use_gpu,
                                         cuda_device=self.gpu_id, num_labels=self.num_labels, args=self.args)

        model.train_model(train_df, verbose=verbose)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.args["output_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.args["output_dir"])

        return self

    def predict(
        self,
        X: pd.Series,
        batch_size: int = 1,
    ):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        batch_size : int, default=1
            Size of batch.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions.
        """
        validate_text_input(X)
        device = 0 if self.use_gpu else -1
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, batch_size=batch_size, device=device, truncation=True)

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

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        validate_text_input(X)
        device = 0 if self.use_gpu else -1

        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True,
                                              batch_size=batch_size, device=device, truncation=True)
        predictions = pipeline(X.tolist())

        probs = []
        for prediction in predictions:
            scores = []
            for label in prediction:
                scores.append(label["score"])

            probs.append(scores)

        return np.ndarray(scores)

    def score(self, X: pd.Series, y: pd.Series, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : pd.Series
            Pandas text series containing targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def save_model(self, path="./save"):
        """
        Save model to disk.

        Parameters
        ----------
        path : str
            Save path.
        """
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        os.makedirs(path, exist_ok=True)

        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path="./save"):
        """
        Load model from disk.

        Parameters
        ----------
        path : str
            Path to model.
        """

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
