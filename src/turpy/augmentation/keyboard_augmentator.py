import pkg_resources
import functools
import pandas as pd
from sklearn.base import TransformerMixin
import nlpaug.augmenter.char as nac


def _duplicator(val, n):
    return [val for _ in range(n)]

class KeyboardAugmentator(TransformerMixin, nac.KeyboardAug):
    def __init__(self,
                 aug_char_min=1,
                 aug_char_max=10,
                 aug_char_p=0.3,
                 aug_word_p=0.3,
                 aug_word_min=1,
                 aug_word_max=10,
                 stopwords=None,
                 tokenizer=None,
                 reverse_tokenizer=None,
                 include_special_char=False,
                 include_numeric=False,
                 include_upper_case=False,
                 lang='tr',
                 verbose=0,
                 stopwords_regex=None,
                 model_path=None,
                 min_char=2
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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, n=5):
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
