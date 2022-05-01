from symspellpy import SymSpell
import pkg_resources

sym_spell = SymSpell()
path = pkg_resources.resource_filename('turpy', 'resources/word_count.txt')
sym_spell.load_dictionary(path, 0, 1, encoding="utf-8")


__all__ = ["correct_sentence", "correct_noisy_sentence", "correct_word"]


def correct_sentence(text: str, max_edit_distance=1, ignore_non_words=False,
                     transfer_casing=False, split_by_space=False,
                     ignore_term_with_digits=False,
                     symspell_cls=None):
    """Correct a sentence."""

    if symspell_cls:
        suggestions = symspell_cls.lookup_compound(text, max_edit_distance=max_edit_distance, ignore_non_words=ignore_non_words,
                                                   transfer_casing=transfer_casing, split_by_space=split_by_space,
                                                   ignore_term_with_digits=ignore_term_with_digits)
    else:
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=max_edit_distance, ignore_non_words=ignore_non_words,
                                                transfer_casing=transfer_casing, split_by_space=split_by_space,
                                                ignore_term_with_digits=ignore_term_with_digits)

    corrected = suggestions[0].term
    return corrected


def correct_noisy_sentence(text: str, max_edit_distance=1, max_segmentation_word_length=None,
                           ignore_token=None, symspell_cls=None):
    """Correct a noisy sentence."""
    if symspell_cls:
        suggestions = symspell_cls.word_segmentation(text, max_edit_distance=max_edit_distance,
                                                     max_segmentation_word_length=max_segmentation_word_length, ignore_token=ignore_token)
    else:
        suggestions = sym_spell.word_segmentation(text, max_edit_distance=max_edit_distance,
                                                  max_segmentation_word_length=max_segmentation_word_length, ignore_token=ignore_token)

    corrected = suggestions[0]
    return corrected


def correct_word(text: str, max_edit_distance=1, include_unknown=False,
                 ignore_token=None, transfer_casing=False, symspell_cls=None):
    """Correct a word."""
    if symspell_cls:
        suggestions = symspell_cls.lookup(text)
    else:
        suggestions = sym_spell.lookup(
            text, max_edit_distance=max_edit_distance, include_unknown=include_unknown,
            ignore_token=ignore_token, transfer_casing=transfer_casing
        )

    corrected = suggestions[0].term
    return corrected
