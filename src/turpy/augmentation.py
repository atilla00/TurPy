import string
import random
import numpy as np
import json
from typing import List
import pkg_resources


class KeyboardAugmentator:

    def __init__(self, upper_cased=False, special_chars=False):
        """
        Keyboard augmentation class.

        One edit distance is used for augmentating
        """

        # Chars to ignore when selecting random char.
        self.IGNORE_CHARS = string.punctuation + string.whitespace

        with open(pkg_resources.resource_filename('turpy', 'resources/tr.json')) as file:
            qwerty_maps = json.load(file)

        if upper_cased:
            for key, values in qwerty_maps.items():
                upper_cased_chars = [ch.upper() for ch in values]
                qwerty_maps[key] = list(set(values + upper_cased_chars))

        if not special_chars:
            for key, values in qwerty_maps.items():
                excluded_special_chars = [ch for ch in values if ch not in string.punctuation]
                qwerty_maps[key] = excluded_special_chars

        self.qwerty_maps = qwerty_maps

    def _swap_n_random_char(self, sentence, n=1) -> str:

        # Cast string to list for index based processing.
        sentence = list(sentence.lower())

        if len(sentence) <= n:
            raise ValueError("Number of random swaps can not be equal or higher than length of the sentence.")

        swapped_idx: List[int] = []

        for i in range(n):
            # Select a random idx that is not in ignored list or already swapped idx.
            while True:
                random_idx = np.random.choice([i for i in range(len(sentence))], replace=False)
                if (sentence[random_idx] in self.IGNORE_CHARS) or (random_idx in swapped_idx):
                    continue
                break

            # Swap randomly selected char to random 1 keyboard distance char.
            char = sentence[random_idx]
            replaced_char = random.choice(self.qwerty_maps[char])
            sentence[random_idx] = replaced_char

        return "".join(sentence)

    def augmentate(self, sentence: str, n_mistakes: int = 1) -> str:
        """
        Augmentate a sentence using keyboard distance.

        Args:
            sentence: Sentence to apply augmentation.
            n_mistakes: Number of keyboard misspelling to apply.

        Returns:
            sentence: Augmentated sentence.
        """

        text = self._swap_n_random_char(sentence, n=n_mistakes)

        return text
