import pkg_resources
import functools
import pandas as pd
from sklearn.base import TransformerMixin
import nlpaug.augmenter.char as nac
from .._types import validate_text_input
from ..base import TokenizerFunc
from typing import List, Optional

def _duplicator(val, n):
    return [val for _ in range(n)]

class KeyboardAugmentator(TransformerMixin, nac.KeyboardAug):
    """
    Text augmentation by simulating keyboard errors.

    Parameters
    ----------
    aug_char_p : float, default=0.3
        Percentage of character (per token) will be augmented.

    aug_char_min : int, default=1
        Minimum number of character will be augmented.

    aug_char_max : int, default=10
        Maximum number of character will be augmented.

    aug_word_p : float, default=0.3
        Percentage of word will be augmented.

    aug_word_min : int, default=1
        Minimum number of word will be augmented.

    aug_word_max : int, default=10
        Maximum number of word will be augmented.

    min_char : int, default=2
        If word less than this value, do not draw word for augmentation.

    stopwords : Optional[List[str]], default=None
        ist of words which will be skipped from augment operation.

    tokenizer : TokenizerFunc, default=None
        Customize tokenization process.

    reverse_tokenizer : TokenizerFunc, default=None
        Customize reverse of tokenization process.

    include_special_char : bool, default=False
        Include special character.

    include_numeric : bool, default=False
         If True, numeric character may be included in augmented data.

    include_upper_case : bool, default=False
        If True, upper case character may be included in augmented data.

    lang : str, default='str'
        Indicate built-in language model.

    verbose : int, default=0
        Verbosity level.

    stopwords_regex : Optional[str], default=None
        Regular expression for matching words which will be skipped from augment operation.

    model_path : Optional[str], default=None
        Loading customize model from file system.
    """

    def __init__(self,
                 aug_char_p: float = 0.3,
                 aug_char_min: int = 1,
                 aug_char_max: int = 10,
                 aug_word_p: float = 0.3,
                 aug_word_min: int = 1,
                 aug_word_max: int = 10,
                 min_char: int = 2,
                 stopwords: Optional[List[str]] = None,
                 tokenizer: TokenizerFunc = None,
                 reverse_tokenizer: TokenizerFunc = None,
                 include_special_char: bool = False,
                 include_numeric: bool = False,
                 include_upper_case: bool = False,
                 lang: str = 'tr',
                 verbose: int = 0,
                 stopwords_regex: Optional[str] = None,
                 model_path: Optional[str] = None,
                 ):

        if stopwords is None:
            path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')
            with open(path, 'r', encoding='utf-8') as file:
                stopwords = list(file.read().split("\n"))

        super().__init__(
            name="KeyboardAugmentator",
            aug_char_min=aug_char_min,
            aug_char_max=aug_char_max,
            aug_char_p=aug_char_p,
            aug_word_p=aug_word_p,
            aug_word_min=aug_word_min,
            aug_word_max=aug_word_max,
            stopwords=stopwords,
            tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer,
            include_special_char=include_special_char,
            include_numeric=include_numeric,
            include_upper_case=include_upper_case,
            lang=lang,
            verbose=verbose,
            stopwords_regex=stopwords_regex,
            model_path=model_path,
            min_char=min_char,
        )

    def __str__(self):
        cls_str_list = [f"\t{k} = {v}" for k, v in self.__dict__.items()]
        cls_str = ",\n".join(cls_str_list)
        return f"""{self.name}(\n{cls_str}\n)"""

    def fit(self, X: pd.Series, y=None, **extra_params):
        """Does nothing. Exist for compatibility reasons for sklearn pipelines."""
        return self

    def transform(self, X: pd.Series, y=None, n=1, **extra_params):
        """Augmentate text from given text series.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : Optional[pd.Series]
            None or Pandas text series containing targets. If provided augmented target series returned.

        n : int
            Number of augmentations to apply.

        Returns
        -------
        X_auged : pd.Series
            Augmented text series.

        y_auged : pd.Series or None
            Augmented target series.
        """
        validate_text_input(X)

        X_auged = X.apply(functools.partial(self.augment, n=n)) \
            .apply(pd.Series) \
            .stack() \
            .reset_index(drop=True)

        if y is None:
            return X_auged

        y_auged = y.apply(functools.partial(_duplicator, n=n)) \
            .apply(pd.Series) \
            .stack() \
            .reset_index(drop=True)

        return X_auged, y_auged

    def fit_transform(self, X, y=None, **extra_params):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : Optional[pd.Series]
            None or Pandas text series containing targets. If provided augmented target series returned.

        Returns
        -------
        X_auged : pd.Series
            Augmented text series.

        y_auged : pd.Series or None
            Augmented target series.
        """
        return self.fit(X, y, **extra_params).transform(X, y, **extra_params)
