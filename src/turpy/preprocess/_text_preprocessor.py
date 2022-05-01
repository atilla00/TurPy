import pkg_resources
import pandas as pd
from typing import Set, List, Union, Optional
from .._types import validate_text_input
from .functions import _preprocess_functions
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocesser(BaseEstimator, TransformerMixin):
    """General text preprocessor.

    Parameters
    ----------
    lowercase : bool, default=False
        Lowercase text

    replace_digits : Union[bool, str], default=False
        Replace digits with provided string. Setting this to True will remove digits.

    replace_digits_blocks_only : Union[bool, str], default=False
        Replace block of digits with provided string. Setting this to True will remove block of digits.

    replace_punctuations : Union[bool, str], default=False
        Replace punctuations with provided string. Setting this to True will remove punctuations.

    replace_emojis : Union[bool, str], default=False
        Replace emojis with provided string. Setting this to True will remove emojis.

    remove_diacritics : Union[bool, str], default=False
        Remove diacritics.

    remove_extra_whitespace : Union[bool, str], default=False
        Remove extra white space.

    replace_urls : Union[bool, str], default=False
        Replace urls with provided string. Setting this to True will remove urls.

    replace_html_tags : Union[bool, str], default=False
        Replace html tags with provided string. Setting this to True will remove html tags.

    replace_tags : Union[bool, str], default=False
        Replace tags with provided string. Setting this to True will remove tags.

    replace_stopwords : Union[bool, str], default=False
        Replace stopwrods with provided string. Setting this to True will remove stopwords.

    stopwords : Optional[Set[str]], default=None
        Set of stopwords.

    order : Optional[List[str]], default=None
        Applying preprocessing functions in order according to given list of strings.
        Example: order=["lowercase", "replace_emojis"] will apply lowercase and replace_emojis in order.

        *Note: If order is provided, only the provided functions will apply if they are set to True/to_replace string.
    """

    def __init__(self,
                 lowercase: bool = False,
                 replace_digits: Union[bool, str] = False,
                 replace_digits_blocks_only: Union[bool, str] = False,
                 replace_punctuations: Union[bool, str] = False,
                 replace_emojis: Union[bool, str] = False,
                 remove_diacritics: bool = False,
                 remove_extra_whitespace: bool = False,
                 replace_urls: Union[bool, str] = False,
                 replace_html_tags: Union[bool, str] = False,
                 replace_hashtags: Union[bool, str] = False,
                 replace_tags: Union[bool, str] = False,
                 replace_stopwords: Union[bool, str] = False,
                 stopwords: Optional[Set[str]] = None,
                 order: Optional[List[str]] = None
                 ):

        self.lowercase = lowercase
        self.replace_digits = replace_digits
        self.replace_digits_blocks_only = replace_digits_blocks_only
        self.replace_punctuations = replace_punctuations
        self.remove_diacritics = remove_diacritics
        self.replace_emojis = replace_emojis
        self.remove_extra_whitespace = remove_extra_whitespace
        self.replace_urls = replace_urls
        self.replace_html_tags = replace_html_tags
        self.replace_hashtags = replace_hashtags
        self.replace_tags = replace_tags
        self.replace_stopwords = replace_stopwords
        self.stopwords = stopwords
        self.order = order

        # Set stopwords
        if self.stopwords is None:
            path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')

            with open(path, 'r', encoding='utf-8') as file:
                self.stopwords = set(file.read().split("\n"))

    def fit(self, X: pd.Series, y=None):
        """Does nothing. Exist for compatibility reasons for sklearn pipelines."""
        return self

    def transform(self, X: pd.Series, y=None):
        """Preprocess text from given text series.

        Parameters
        ----------
        X: pd.Series
            Pandas text series containing texts.

        y: Optional[pd.Series]
            Ignored.

        Returns
        ----------
        X: pd.Series
            Preprocessed text series.


        """
        validate_text_input(X)

        if self.order:

            for attr_name in self.order:
                attribute = getattr(self, attr_name)
                func = getattr(_preprocess_functions, attr_name)

                if attribute is False:
                    continue
                elif attribute is True:
                    to_replace = ""
                    X = func(X, to_replace, self.stopwords)
                else:
                    X = func(X, to_replace, self.stopwords)

        else:

            ignore = ["order", "stopwords"]

            for attr_name in self.__dict__.keys():
                if attr_name in ignore:
                    continue

                attribute = getattr(self, attr_name)
                func = getattr(_preprocess_functions, attr_name)

                if attribute is False:
                    continue
                elif attribute is True:
                    to_replace = ""
                    X = func(X, to_replace, self.stopwords)
                else:
                    to_replace = attribute
                    X = func(X, to_replace, self.stopwords)

        return X
