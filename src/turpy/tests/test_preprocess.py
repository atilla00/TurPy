from turpy.preprocess import TextPreprocesser, SpellingPreprocessor
import pandas as pd

def test_lowercase():
    test_input = pd.Series(["AAAAA", "aAaA"])
    expected = pd.Series(["aaaaa", "aaaa"])

    processor = TextPreprocesser(lowercase=True)
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_digits():
    test_input = pd.Series(["telefon numaram +90 123 1232 2312"])

    # Without a replacer
    expected = pd.Series(["telefon numaram +   "])
    processor = TextPreprocesser(replace_digits=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["telefon numaram +x x x x"])
    processor = TextPreprocesser(replace_digits="x")
    assert expected.equals(processor.fit_transform(test_input))

def test_replace_digits_blocks_only():
    test_input = pd.Series(["kayıt numaram KT123456-12"])

    # Without a replacer
    expected = pd.Series(["kayıt numaram KT123456-"])
    processor = TextPreprocesser(replace_digits_blocks_only=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["kayıt numaram KT123456-x"])
    processor = TextPreprocesser(replace_digits_blocks_only="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_punctuations():
    test_input = pd.Series(["sana bi, soru sorabilirmiyi.?"])

    # Without a replacer
    expected = pd.Series(["sana bi soru sorabilirmiyi"])
    processor = TextPreprocesser(replace_punctuations=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["sana bix soru sorabilirmiyixx"])
    processor = TextPreprocesser(replace_punctuations="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_replace_emojis():
    test_input = pd.Series(["hadi len 😂"])

    # Without a replacer
    expected = pd.Series(["hadi len "])
    processor = TextPreprocesser(replace_emojis=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["hadi len x"])
    processor = TextPreprocesser(replace_emojis="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_remove_diacritics():
    test_input = pd.Series(["ğüiİşçö"])
    expected = pd.Series(["guiIsco"])

    processor = TextPreprocesser(remove_diacritics=True)
    assert expected.equals(processor.fit_transform(test_input))


def test_remove_extra_whitespace():
    test_input = pd.Series(["abi\ndük     ddd  dd\n"])
    expected = pd.Series(["abi dük ddd dd"])

    processor = TextPreprocesser(remove_extra_whitespace=True)
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_urls():
    test_input = pd.Series(["https://www.github.com/ yükle kodunu"])

    # Without a replacer
    expected = pd.Series([" yükle kodunu"])
    processor = TextPreprocesser(replace_urls=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["x yükle kodunu"])
    processor = TextPreprocesser(replace_urls="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_html_tags():
    test_input = pd.Series(["<h1>Başlık</h1>"])

    # Without a replacer
    expected = pd.Series(["Başlık"])
    processor = TextPreprocesser(replace_html_tags=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["xBaşlıkx"])
    processor = TextPreprocesser(replace_html_tags="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_hashtags():
    test_input = pd.Series(["#barajas_dragon napıyon"])

    # Without a replacer
    expected = pd.Series([" napıyon"])
    processor = TextPreprocesser(replace_hashtags=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["x napıyon"])
    processor = TextPreprocesser(replace_hashtags="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_tags():
    test_input = pd.Series(["@barajasdragon napıyon"])

    # Without a replacer
    expected = pd.Series([" napıyon"])
    processor = TextPreprocesser(replace_tags=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a replacer
    expected = pd.Series(["x napıyon"])
    processor = TextPreprocesser(replace_tags="x")
    assert expected.equals(processor.fit_transform(test_input))


def test_replace_stopwords():
    test_input = pd.Series(["ya napıyon", "bu tavırlara karşıyım ammavelakin sen yaparsan olur"])

    # Without a provided stopwords
    expected = pd.Series([" napıyon", " tavırlara karşıyım ammavelakin sen yaparsan olur"])
    processor = TextPreprocesser(replace_stopwords=True)
    assert expected.equals(processor.fit_transform(test_input))

    # With a provided stopwords
    stopwords = set(["ammavelakin"])
    expected = pd.Series(["ya napıyon", "bu tavırlara karşıyım x sen yaparsan olur"])
    processor = TextPreprocesser(replace_stopwords="x", stopwords=stopwords)
    assert expected.equals(processor.fit_transform(test_input))


def test_spellchecker():
    test_input = pd.Series(["ne oluyr ısteme", "herhaldeyapmazonlar"])

    expected = pd.Series(["ne olur isteme", "herhalde amazonlar"])  # Totally wrong
    speller = SpellingPreprocessor(speller="sentence", max_edit_distance=1)
    assert expected.equals(speller.fit_transform(test_input))

    expected = pd.Series(["ne oluyr ısteme", "herhalde yapmaz onlar"])
    speller = SpellingPreprocessor(speller="noisy_sentence", max_edit_distance=1)
    assert expected.equals(speller.fit_transform(test_input))
