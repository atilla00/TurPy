Functions
==============================

Functions to apply directly on text or text series.

::

    from turpy.preprocess.functions import *

    correct_sentence("yazm hatası")
    remove_diacritics(pd.Series(["iğüçö"]))


List of available functions
-----------------------------


**Spelling Preprocess**:

* correct_sentence
* correct_noisy_sentence
* correct_word

**Text Preprocess**:

* lowercase
* replace_digits
* replace_digits_blocks_only
* replace_punctuations
* remove_diacritics
* replace_html_tags
* replace_hashtags
* replace_tags
* replace_stopwords
* replace_emojis
* remove_extra_whitespace