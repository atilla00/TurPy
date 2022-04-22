import pkg_resources
import pandas as pd
from tqdm import tqdm
from symspellpy import SymSpell
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from .._types import validate_text_input
tqdm.pandas()

class SpellingPreprocessor(BaseEstimator, TransformerMixin):
    """Spelling preprocessor using SymSpell.

    Parameters
    ----------
    speller : str, default='sentence'
        Which algorithms to use for correction. Possible algorithms: ['sentence', 'noisy_sentence'].

    max_edit_distance : int, default=1
        The maximum edit distance between input andsuggested words.

    word_counts_file : Optional[str], default=None
        Load word count file from filesystem. Default uses from Turkish TurPy word counts file.

    ignore_non_words : bool, default=False
        A flag to determine whether numbers and acronyms are left alone during the spell checking process.

    transfer_casing : bool, default=False
        A flag to determine whether the casing --- i.e., uppercase vs lowercase --- should be carried over from `phrase`.

    split_by_space : bool, default=False
        Splits the phrase into words simply based on space.

    ignore_term_with_digits : bool, default=False
        A flag to determine whether any term with digits is left alone during the spell checking process. Only works when ``ignore_non_words` is also ``True``.

    max_segmentation_word_length : Optional[int], default=None
        The maximum word length that should be considered.

    ignore_token : Optional[str], default=None
        A regex pattern describing what words/phrases to ignore and leave unchanged.
    """

    def __init__(self,
                 speller: str = "sentence",
                 max_edit_distance: int = 1,
                 word_counts_file: Optional[str] = None,

                 # Sentence
                 ignore_non_words: bool = False,
                 transfer_casing: bool = False,
                 split_by_space: bool = False,
                 ignore_term_with_digits: bool = False,

                 # Noisy Sentence
                 max_segmentation_word_length: Optional[int] = None,
                 ignore_token: Optional[str] = None,
                 ):

        self.speller = speller
        self.max_edit_distance = max_edit_distance
        self.word_counts_file = word_counts_file

        self.ignore_non_words = ignore_non_words
        self.transfer_casing = transfer_casing
        self.split_by_space = split_by_space
        self.ignore_term_with_digits = ignore_term_with_digits

        self.max_segmentation_word_length = max_segmentation_word_length
        self.ignore_token = ignore_token

        if self.speller not in ["sentence", "noisy_sentence"]:
            raise ValueError("Speller can only be one of the ['sentence', 'noisy_sentence']")

        self.sym_spell = SymSpell()

        if self.word_counts_file:
            self.sym_spell.load_dictionary(self.word_counts_file, 0, 1, encoding="utf-8")
        else:
            path = pkg_resources.resource_filename('turpy', 'resources/word_count.txt')
            self.sym_spell.load_dictionary(path, 0, 1, encoding="utf-8")

    def _correct_spelling(self, text: str):
        """Correct a string"""
        if self.speller == "sentence":
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=self.max_edit_distance, ignore_non_words=self.ignore_non_words,
                                                         transfer_casing=self.transfer_casing, split_by_space=self.split_by_space,
                                                         ignore_term_with_digits=self.ignore_term_with_digits)
            corrected = suggestions[0].term

        elif self.speller == "noisy_sentence":
            suggestions = self.sym_spell.word_segmentation(text, max_edit_distance=self.max_edit_distance,
                                                           max_segmentation_word_length=self.max_segmentation_word_length, ignore_token=self.ignore_token)
            corrected = suggestions[0]

        # elif self.speller == "word":
        #    suggestions = self.sym_spell.lookup(text)
        #    corrected = suggestions[0].term

        return corrected

    def fit(self, X: pd.Series, y=None):
        """Does nothing. Exist for compatibility reasons for sklearn pipelines."""
        return self

    def transform(self, X: pd.Series, y=None):
        """Preprocess text from given text series.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : Optional[pd.Series]
            Ignored.

        Returns
        -------
        X : pd.Series
            Preprocessed text series.
        """
        validate_text_input(X)
        X = X.progress_apply(self._correct_spelling)

        return X
