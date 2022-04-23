import re
import string
import unicodedata
import pkg_resources
import pandas as pd
from typing import Set, Union, Optional
from .._types import validate_text_input
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
                 stopwords: Optional[Set[str]] = None
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

    def _do_lowercase(self, s, to_replace):
        validate_text_input(s)
        return s.str.lower()

    def _do_replace_digits(self, s, to_replace):
        validate_text_input(s)
        return s.str.replace(r"\d+", to_replace, regex=True)

    def _do_replace_digits_blocks_only(self, s, to_replace):
        validate_text_input(s)
        return s.str.replace(r"\b\d+\b", to_replace, regex=True)

    def _do_replace_punctuations(self, s, to_replace):
        validate_text_input(s)
        return s.str.replace(rf"([{string.punctuation}])", to_replace, regex=True)

    def __do_remove_diacritics(self, text):
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    def _do_remove_diacritics(self, s, to_replace):
        validate_text_input(s)
        return s.astype("unicode").apply(self.__do_remove_diacritics)

    def _do_replace_urls(self, s, to_replace):
        validate_text_input(s)
        return s.str.replace(r"http\S+", to_replace, regex=True)

    def _do_replace_html_tags(self, s, to_replace):
        validate_text_input(s)
        pattern = r"""(?x)                              # Turn on free-spacing
        <[^>]+>                                       # Remove <html> tags
        | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
        """

        return s.str.replace(pattern, to_replace, regex=True)

    def _do_replace_hashtags(self, s, to_replace):
        validate_text_input(s)
        pattern = r"#[a-zA-Z0-9_]+"
        return s.str.replace(pattern, to_replace, regex=True)

    def _do_replace_tags(self, s, to_replace):
        validate_text_input(s)
        pattern = r"@[a-zA-Z0-9_]+"
        return s.str.replace(pattern, to_replace, regex=True)

    def __do_replace_stopwords(self, text, stopwords, to_replace):
        pattern = r"""(?x)                          # Set flag to allow verbose regexps
        \w+(?:-\w+)*                              # Words with optional internal hyphens
        | \s*                                     # Any space
        | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol
        """
        return "".join(t if t not in stopwords else to_replace for t in re.findall(pattern, text))

    def _do_replace_stopwords(self, s, to_replace):
        validate_text_input(s)
        if self.stopwords is None:
            path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')

            with open(path, 'r', encoding='utf-8') as file:
                self.stopwords = set(file.read().split("\n"))

        return s.apply(self.__do_replace_stopwords, args=(self.stopwords, to_replace))

    def _do_replace_emojis(self, s, to_replace):
        validate_text_input(s)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        return s.str.replace(emoji_pattern, to_replace, regex=True)

    def _do_remove_extra_whitespace(self, s, to_replace):
        validate_text_input(s)
        return s.str.replace("\xa0", " ").str.split().str.join(" ")

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
        # Automatic
        # class_methods = [method_name for method_name in dir(self)
        #                 if method_name.startswith("do_") if callable(getattr(self, method_name))]

        # Ordered
        class_methods = [
            "_do_lowercase", "_do_remove_diacritics", "_do_replace_punctuations", "_do_replace_emojis",
            "_do_replace_digits", "_do_replace_digits_blocks_only", "_do_replace_urls", "_do_replace_html_tags", "_do_replace_hashtags", "_do_replace_tags",
            "_do_replace_stopwords", "_do_remove_extra_whitespace"
        ]

        for method_name in class_methods:
            # Ignore "do_" prefix.
            attribute_name = method_name[4:]

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
