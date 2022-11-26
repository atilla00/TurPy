import re
import string
import unicodedata

import pandas as pd

from turpy.preprocess._base import BaseTextTransformer

__all__ = ["LowerCaseTransformer", "ExtraWhiteSpaceTransformer", "DiacriticTransformer",
     "DigitTransformer", "PunctuationTransformer", "UrlTransformer", "HtmlTagTransformer", 
     "HashtagTransformer", "EmojiTransformer", "TagTransformer"]

class LowerCaseTransformer(BaseTextTransformer):
    def transform(self, X: pd.Series, y: pd.Series = None):
        return X.str.replace("I", "ı").str.lower()

class ExtraWhiteSpaceTransformer(BaseTextTransformer):
    def transform(self, X: pd.Series, y: pd.Series = None):
        return X.str.replace("\xa0", " ").str.split().str.join(" ")

class DiacriticTransformer(BaseTextTransformer):
    @staticmethod
    def _remove_diacritics(text: str) -> str:
        text = text.replace("ı", "i")
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    def transform(self, X: pd.Series, y: pd.Series = None):
        return X.astype("unicode").apply(self._remove_diacritics)


class DigitTransformer(BaseTextTransformer):
    def __init__(self, to_replace: str = "", blocks_only: bool = False):
        self.to_replace = to_replace
        self.blocks_only = blocks_only

        if self.blocks_only:
            self.pattern = r"\b[-+]?(?:\d*\.\d+|\d+)\b"
        else:
            self.pattern = r"[-+]?(?:\d*\.\d+|\d+)" #r"\d+"


class PunctuationTransformer(BaseTextTransformer):
    pattern = rf"([{string.punctuation}])"


class UrlTransformer(BaseTextTransformer):
    pattern = r"http\S+"


class HtmlTagTransformer(BaseTextTransformer):
    pattern = r"""(?x)                               # Turn on free-spacing
                <[^>]+>                                       # Remove <html> tags
                | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
                """


class HashtagTransformer(BaseTextTransformer):
    pattern = r"#[a-zA-Z0-9_]+"


class TagTransformer(BaseTextTransformer):
    pattern = r"#[a-zA-Z0-9_]+"


class EmojiTransformer(BaseTextTransformer):
    pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
