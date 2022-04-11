import re
import string
import unicodedata
import pkg_resources
from typing import Set, Union
from .._types import check_input
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocesser(BaseEstimator, TransformerMixin):
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
                 stopwords: Union[Set[str], None] = None
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

    def do_lowercase(self, s, to_replace):
        check_input(s)
        return s.str.lower()

    def do_replace_digits(self, s, to_replace):
        check_input(s)
        return s.str.replace(r"\d+", to_replace, regex=True)

    def do_replace_digits_blocks_only(self, s, to_replace):
        check_input(s)
        return s.str.replace(r"\b\d+\b", to_replace, regex=True)

    def do_replace_punctuations(self, s, to_replace):
        check_input(s)
        return s.str.replace(rf"([{string.punctuation}])+", to_replace, regex=True)

    def _do_remove_diacritics(self, text):
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    def do_remove_diacritics(self, s, to_replace):
        check_input(s)
        return s.astype("unicode").apply(self._do_remove_diacritics)

    def do_replace_urls(self, s, to_replace):
        check_input(s)
        return s.str.replace(r"http\S+", to_replace, regex=True)

    def do_replace_html_tags(self, s, to_replace):
        check_input(s)
        pattern = r"""(?x)                              # Turn on free-spacing
        <[^>]+>                                       # Remove <html> tags
        | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
        """

        return s.str.replace(pattern, to_replace, regex=True)

    def do_replace_hashtags(self, s, to_replace):
        check_input(s)
        pattern = r"#[a-zA-Z0-9_]+"
        return s.str.replace(pattern, to_replace, regex=True)

    def do_replace_tags(self, s, to_replace):
        check_input(s)
        pattern = r"@[a-zA-Z0-9_]+"
        return s.str.replace(pattern, to_replace, regex=True)

    def _do_replace_stopwords(self, text, stopwords, to_replace):
        pattern = r"""
                    (?x)
                    \w+(?:-\w+)*
                    | \s*
                    | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]
                    """
        return "".join(t if t not in stopwords else to_replace for t in re.findall(pattern, text))

    def do_replace_stopwords(self, s, to_replace):
        check_input(s)
        if self.stopwords is None:
            path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')

            with open(path, 'r', encoding='utf-8') as file:
                stopwords = set(file.read().split("\n"))

        return s.apply(self._do_replace_stopwords, args=(stopwords, to_replace))

    def do_replace_emojis(self, s, to_replace):
        check_input(s)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        return s.str.replace(emoji_pattern, to_replace, regex=True)

    def do_remove_extra_whitespace(self, s, to_replace):
        check_input(s)
        return s.str.replace("\xa0", " ").str.split().str.join(" ")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        check_input(X)
        # Automatic
        # class_methods = [method_name for method_name in dir(self)
        #                 if method_name.startswith("do_") if callable(getattr(self, method_name))]

        # Ordered
        class_methods = [
            "do_lowercase", "do_remove_diacritics", "do_replace_punctuations", "do_replace_emojis",
            "do_replace_digits", "do_replace_digits_blocks_only", "do_replace_urls", "do_replace_html_tags", "do_replace_hashtags", "do_replace_tags",
            "do_replace_stopwords", "do_remove_extra_whitespace"
        ]

        for method_name in class_methods:
            # Ignore "do_" prefix.
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
