from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


def make_wordcloud(text_series: pd.Series, wordcloud_settings=None):

    if wordcloud_settings is None:
        wordcloud = WordCloud(max_font_size=100, random_state=42, width=1000, height=860, margin=2)
    else:
        wordcloud = WordCloud(**wordcloud_settings)

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    return fig
