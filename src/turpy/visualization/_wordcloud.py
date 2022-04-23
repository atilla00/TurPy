from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Optional, Dict
import pandas as pd
from .._types import validate_text_input


def make_wordcloud(text_series: pd.Series, wordcloud_settings: Optional[Dict[str, str]] = None):
    """Plot a WordCloud from given text series.

    Parameters
    -------------
    text_series : pd.Series
        Pandas text series containing texts.

    word_cloudsettings : Optional[dict]
        WordCloud settings in text format.


    Returns
    ----------
    fig : plt.Figure
        Matplotlib figure.


    """
    validate_text_input(text_series)

    if wordcloud_settings is None:
        wordcloud = WordCloud(max_font_size=100, random_state=42, width=1000, height=860, margin=2)
    else:
        wordcloud = WordCloud(**wordcloud_settings)

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    return fig
