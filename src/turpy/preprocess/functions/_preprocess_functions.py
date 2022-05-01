
import unicodedata
import pandas as pd
import string
import re
from ..._types import TextSeries
from typing import Set


__all__ = [
    "lowercase", "replace_digits", "replace_digits_blocks_only", "replace_punctuations", "remove_diacritics", "replace_urls",
    "replace_html_tags", "replace_hashtags", "replace_tags", "replace_stopwords", "replace_emojis", "remove_extra_whitespace"
]

@TextSeries
def lowercase(s: pd.Series, *args) -> pd.Series:
    """Lowercase a text series."""
    return s.str.replace("I", "ı").str.lower()


@TextSeries
def replace_digits(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace digits in a text series."""
    return s.str.replace(r"\d+", to_replace, regex=True)


@TextSeries
def replace_digits_blocks_only(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace blocks of digits in a text series."""
    return s.str.replace(r"\b\d+\b", to_replace, regex=True)


@TextSeries
def replace_punctuations(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace punctuations in a text series."""
    return s.str.replace(rf"([{string.punctuation}])", to_replace, regex=True)


@TextSeries
def remove_diacritics(s: pd.Series, *args) -> pd.Series:
    """Remove diacritics in a text series."""
    return s.astype("unicode").apply(_remove_diacritics)


@TextSeries
def replace_urls(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace urls in a text series."""
    return s.str.replace(r"http\S+", to_replace, regex=True)


@TextSeries
def replace_html_tags(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace html tags in text series."""
    pattern = r"""(?x)                              # Turn on free-spacing
    <[^>]+>                                       # Remove <html> tags
    | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
    """
    return s.str.replace(pattern, to_replace, regex=True)


@TextSeries
def replace_hashtags(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace hashtags in text series."""
    pattern = r"#[a-zA-Z0-9_]+"
    return s.str.replace(pattern, to_replace, regex=True)


@TextSeries
def replace_tags(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace tags in atext series."""
    pattern = r"@[a-zA-Z0-9_]+"
    return s.str.replace(pattern, to_replace, regex=True)


@TextSeries
def replace_stopwords(s: pd.Series, to_replace: str, stopwords: Set[str], *args) -> pd.Series:
    """Replace stopwords in a text series with given set of stopwords."""
    return s.apply(_replace_stopwords, args=(stopwords, to_replace))


@TextSeries
def replace_emojis(s: pd.Series, to_replace: str, *args) -> pd.Series:
    """Replace emojis in a text series."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    return s.str.replace(emoji_pattern, to_replace, regex=True)


@TextSeries
def remove_extra_whitespace(s: pd.Series, *args) -> pd.Series:
    """Remove extra white space in a text series"""
    return s.str.replace("\xa0", " ").str.split().str.join(" ")


# Utilities
def _replace_stopwords(text: str, stopwords: Set[str], to_replace: str) -> str:
    """Replace stopwords in a string with given set of stopwords."""
    pattern = r"""(?x)                          # Set flag to allow verbose regexps
    \w+(?:-\w+)*                              # Words with optional internal hyphens
    | \s*                                     # Any space
    | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol
    """
    return "".join(t if t not in stopwords else to_replace for t in re.findall(pattern, text))


def _remove_diacritics(text: str) -> str:
    """Remove diacritics in a string."""
    text = text.replace("ı", "i")
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])
