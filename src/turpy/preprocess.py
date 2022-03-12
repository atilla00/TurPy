"""
TurPy Preprocessing library is mostly consist of modified Texthero functions.

https://github.com/jbesomi/texthero/blob/master/texthero/preprocessing.py
"""

import re
import string
import unicodedata
import pkg_resources
import pandas as pd
from typing import Set, Union, Pattern, Optional
from symspellpy import SymSpell, Verbosity
from sklearn.base import BaseEstimator, TransformerMixin
from functools import lru_cache
from tqdm import tqdm
tqdm.pandas()

def check_types(s):

    if not isinstance(s, pd.Series):
        raise ValueError("Input should be pandas series.")

    try:
        first_non_nan_value = s.loc[s.first_valid_index()]
        if not isinstance(first_non_nan_value, str):
            raise ValueError("Pandas series should consist of only text.")
    except KeyError:  # Only NaNs in Series -> same warning applies
        raise ValueError("Pandas series should consist of only text.")


class SpellingPreprocessor(BaseEstimator, TransformerMixin):
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

    def correct_spelling(self, text):

        if self.speller == "sentence":
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=self.max_edit_distance, ignore_non_words=self.ignore_non_words,
                                                         transfer_casing=self.transfer_casing, split_by_space=self.split_by_space, ignore_term_with_digits=self.ignore_term_with_digits)
            corrected = suggestions[0].term

        elif self.speller == "noisy_sentence":
            suggestions = self.sym_spell.word_segmentation(text, max_edit_distance=self.max_edit_distance,
                                                           max_segmentation_word_length=self.max_segmentation_word_length, ignore_token=self.ignore_token)
            corrected = suggestions[0]

        elif self.speller == "word":
            suggestions = self.sym_spell.lookup(text)
            corrected = suggestions[0].term

        return corrected

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        check_types(X)

        if self.speller == "word":
            raise Exception("Word spell checker is not implemented for Scikit-learn API. Use 'correct_spelling(word)' for single word correction")

        X = X.progress_apply(self.correct_spelling)

        return X

class TextPreprocesser(BaseEstimator, TransformerMixin):
    def __init__(self,
                 lowercase: Union[bool] = False,
                 replace_digits: Union[bool, str] = False,
                 replace_digits_blocks_only: Union[bool, str] = False,
                 replace_punctuations: Union[bool, str] = False,
                 remove_diacritics: Union[bool] = False,
                 remove_extra_whitespace: Union[bool] = False,
                 replace_urls: Union[bool, str] = False,
                 replace_stopwords: Union[bool, str] = False,
                 stopwords: Union[Set[str], None] = None
                 ):

        self.lowercase = lowercase
        self.replace_digits = replace_digits
        self.replace_digits_blocks_only = replace_digits_blocks_only
        self.replace_punctuations = replace_punctuations
        self.remove_diacritics = remove_diacritics
        self.remove_extra_whitespace = remove_extra_whitespace
        self.replace_urls = replace_urls
        self.replace_stopwords = replace_stopwords
        self.stopwords = stopwords

    def check_types(self, s):

        if not isinstance(s, pd.Series):
            raise ValueError("Input should be pandas series.")

        try:
            first_non_nan_value = s.loc[s.first_valid_index()]
            if not isinstance(first_non_nan_value, str):
                raise ValueError("Pandas series should consist of only text.")
        except KeyError:  # Only NaNs in Series -> same warning applies
            raise ValueError("Pandas series should consist of only text.")

    def do_lowercase(self, s, to_replace):
        return s.str.lower()

    def do_replace_digits(self, s, to_replace):
        return s.str.replace(r"\d+", to_replace, regex=True)

    def do_replace_digits_blocks_only(self, s, to_replace):
        return s.str.replace(r"\b\d+\b", to_replace, regex=True)

    def do_replace_punctuations(self, s, to_replace):
        return s.str.replace(rf"([{string.punctuation}])+", to_replace, regex=True)

    def _do_remove_diacritics(self, text):
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    def do_remove_diacritics(self, s, to_replace):
        return s.astype("unicode").apply(self._do_remove_diacritics)

    def do_replace_urls(self, s, to_replace):
        return s.str.replace(r"http\S+", to_replace, regex=True)

    def _do_replace_stopwords(self, text, stopwords, to_replace):
        pattern = r"""
                    (?x)
                    \w+(?:-\w+)*
                    | \s*
                    | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]
                    """
        return "".join(t if t not in stopwords else to_replace for t in re.findall(pattern, text))

    def do_replace_stopwords(self, s, to_replace):
        if self.stopwords is None:
            path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')

            with open(path, 'r', encoding='utf-8') as file:
                stopwords = set(file.read().split("\n"))

        return s.apply(self._do_replace_stopwords, args=(stopwords, to_replace))

    def do_remove_extra_whitespace(self, s, to_replace):
        return s.str.replace("\xa0", " ").str.split().str.join(" ")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Check type of the data
        check_types(X)

        # Automatic
        # class_methods = [method_name for method_name in dir(self)
        #                 if method_name.startswith("do_") if callable(getattr(self, method_name))]

        # Ordered
        class_methods = [
            "do_lowercase", "do_remove_diacritics", "do_replace_punctuations",
            "do_replace_digits", "do_replace_digits_blocks_only", "do_replace_urls",
            "do_replace_stopwords", "do_remove_extra_whitespace"
        ]

        for method_name in class_methods:
            attribute_name = method_name[3:]

            attribute = getattr(self, attribute_name)
            func = getattr(self, method_name)

            if attribute is False:
                continue
            elif attribute is True:
                to_replace = ""
                X = func(X, to_replace)
            else:
                X = func(X, attribute)

        return X
