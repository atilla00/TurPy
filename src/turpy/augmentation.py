import string
import random
import numpy as np
import json
from typing import List, Dict
import pkg_resources


def load_qwerty_mappings() -> Dict[str, List[str]]:
    """
    Loads Turkish qwerty mappings from resources.
    """
    qwerty_maps: Dict[str, List[str]]

    with open(pkg_resources.resource_filename('turpy', 'resources/qwerty_maps.json'), encoding='utf-8') as file:
        qwerty_maps = json.load(file)

    all_chars = list("abcçdefgğhıijklmnoöprsştuüvyzqw")

    # Add upper cased chars to qwerty_maps keys.
    for char in all_chars:
        upper_char = char.upper()
        qwerty_maps[upper_char] = qwerty_maps[char]

    return qwerty_maps


class KeyboardAugmentator:

    """
    Keyboard augmentation class.

    One edit keyboard distance is used for augmentating sentence.

    Args:
        upper_cased: Simulate upper-case spelling mistakes.
        special_chars: Simulate special characters spelling mistakes.
        numbers: Simulate numeric spelling mistakes.
    """

    def __init__(self, upper_cased=False, special_chars=False, numbers=True):

        # Chars to ignore when selecting random char.
        self.IGNORE_CHARS = string.punctuation + string.whitespace

        qwerty_maps = load_qwerty_mappings()

        if upper_cased:
            for key, values in qwerty_maps.items():
                upper_cased_chars = [ch.upper() for ch in values]
                qwerty_maps[key] = list(set(values + upper_cased_chars))

        if not special_chars:
            for key, values in qwerty_maps.items():
                special_chars_excluded = [ch for ch in values if ch not in string.punctuation]
                qwerty_maps[key] = special_chars_excluded

        if not numbers:
            for key, values in qwerty_maps.items():
                numbers_excluded = [ch for ch in values if ch.isdigit()]
                qwerty_maps[key] = numbers_excluded

        self.qwerty_maps = qwerty_maps

    def _swap_n_random_char(self, sentence: str, n_mistakes=1) -> str:
        """
        Swaps n random char from given a sentence.
        """
        # Cast string to list for index based processing.
        char_list = list(sentence)

        if len(char_list) <= n_mistakes:
            raise ValueError("Number of random swaps can not be equal or higher than length of the sentence.")

        char_idxs = [idx for idx, char in enumerate(char_list) if char not in self.IGNORE_CHARS]

        random_idxs = np.random.choice(char_idxs, size=n_mistakes, replace=False)

        for idx in random_idxs:
            char = char_list[idx]
            char_list[idx] = random.choice(self.qwerty_maps[char])

        return "".join(char_list)

    def augmentate(self, sentence: str, n_mistakes: int = 1) -> str:
        """
        Augmentate a sentence using keyboard distance.

        Args:
            sentence: Sentence to apply augmentation.
            n_mistakes: Number of keyboard misspelling to apply.

        Returns:
            sentence: Augmentated sentence.
        """

        text = self._swap_n_random_char(sentence, n_mistakes=n_mistakes)

        return text

    def fit(self):
        return self
