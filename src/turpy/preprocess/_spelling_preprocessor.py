import pkg_resources
from tqdm import tqdm
from symspellpy import SymSpell
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from .._types import _check_types
tqdm.pandas()

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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        _check_types(X)
        X = X.progress_apply(self.correct_spelling)

        return X
