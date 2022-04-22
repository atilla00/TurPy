from turpy.augmentation import EDAAugmentator, KeyboardAugmentator
import pandas as pd


def test_eda_augmentator():
    test_input = pd.Series(["bu tavırlara karşıyım ammavelakin sen yaparsan olur"])

    aug = EDAAugmentator(max_augment=10)
    output = aug.fit_transform(test_input)
    assert len(output) <= aug.max_augment

    test_input_y = pd.Series([1])
    output_X, output_y = aug.fit_transform(test_input, test_input_y)
    assert output_X is not None
    assert output_y is not None
    assert len(output_X) == len(output_y)
    assert len(output_y.drop_duplicates()) == 1


def test_keyboard_augmentator():
    test_input = pd.Series(["hiçbir kelime yedi harfi geçmez"])

    aug = KeyboardAugmentator(min_char=7)
    num_aug = 5
    output = aug.fit_transform(test_input, n=num_aug)
    assert len(output) == num_aug

    test_input_y = pd.Series([1])
    aug = KeyboardAugmentator(min_char=7)
    num_aug = 5
    output_X, output_y = aug.fit_transform(test_input, test_input_y, n=num_aug)
    assert len(output_X) == num_aug
    assert output_X is not None
    assert output_y is not None
    assert len(output_X) == len(output_y)
    assert len(output_y.drop_duplicates()) == 1
