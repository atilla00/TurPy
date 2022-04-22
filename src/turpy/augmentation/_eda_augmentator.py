# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou


import functools
from sklearn.base import BaseEstimator, TransformerMixin
from WordNet.WordNet import WordNet
from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer
from Dictionary.TxtWord import TxtWord

import re
import random
import pandas as pd
import pkg_resources
from typing import List
from random import shuffle
from tqdm import tqdm
from .._types import validate_text_input
tqdm.pandas()
random.seed(1)

def _duplicator(val, n):
    return [val for _ in range(int(n))]

def _duplicator_pandas(row):
    """duplicate {row[0]} as {row[1]} times"""
    return _duplicator(row[0], row[1])


wordnet = WordNet()
fsm = FsmMorphologicalAnalyzer(fileName=pkg_resources.resource_filename('turpy', 'resources/turkish_finite_state_machine.xml'))

# stop words list
path = pkg_resources.resource_filename('turpy', 'resources/stopwords.txt')
with open(path, 'r', encoding='utf-8') as file:
    stopwords = list(file.read().split("\n"))


def find_synonyms(word):
    synonym_set = set()

    # Check if word exist in the word list already
    if word in wordnet.literalList():
        root = word
    else:
        fsmParseList = fsm.morphologicalAnalysis(word)
        root = fsmParseList.getFsmParse(0).root.getName()

    for synset in wordnet.getSynSetsWithLiteral(root):
        synonyms = synset.getSynonym()

        for i in range(synonyms.literalSize()):
            name = synonyms.getLiteral(i).getName()
            synonym_set.add(name)

    if root in synonym_set:
        synonym_set.remove(root)

    return list(synonym_set)


def add_original_word_suffixes(original, synonym):
    """original text, synonym(root)"""
    fsmParse = fsm.morphologicalAnalysis(original)
    original_parsed = fsmParse.getFsmParse(0)

    # First check if word roots are same (pronoun, verb, adjective, adverb) else return empty string

    # One way is this
    original_root = original_parsed.root
    synonym_root = fsm.morphologicalAnalysis(synonym).getFsmParse(0).root

    # If original is already a root
    if original in wordnet.literalList():
        return synonym

    word_types = ['isAbbreviation', 'isAdjective', 'isAdverb', 'isConjunction',
                  'isDate', 'isDeterminer', 'isDuplicate', 'isExceptional', 'isFraction', 'isHeader',
                  'isInterjection', 'isNominal', 'isNumeral', 'isOrdinal',
                  'isPassive', 'isPercent', 'isPlural', 'isPortmanteau', 'isPortmanteauEndingWithSI',
                  'isPortmanteauFacedSoftening', 'isPortmanteauFacedVowelEllipsis', 'isPostP', 'isPronoun', 'isProperNoun',
                  'isPunctuation', 'isPureAdjective', 'isQuestion', 'isRange', 'isReal', 'isSuffix',
                  'isTime', 'isVerb']

    did_true_matched = False

    for word_type in word_types:
        match_root = getattr(original_root, word_type)()
        match_synonym = getattr(synonym_root, word_type)()

        if match_root and match_synonym:
            did_true_matched = True

    if not did_true_matched:
        return ''

    added = fsm.replaceRootWord(parse=original_parsed, newRoot=TxtWord(name=synonym))

    return added

# cleaning up text
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnmıüğşöç ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):

    output: List[str] = []

    try:
        synonyms = find_synonyms(word)
    except IndexError:
        return output

    for syn in synonyms:
        try:
            output.append(add_original_word_suffixes(word, syn))
        except IndexError:
            continue

    return output

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms: List[str] = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1

    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    # rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    # rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[: num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    # augmented_sentences.append(sentence)

    augmented_senteces = list(set(augmented_sentences))
    augmented_senteces = [sentence.strip() for sentence in augmented_senteces]

    return augmented_senteces


class EDAAugmentator(BaseEstimator, TransformerMixin):
    r"""Easy Data Augmentation (EDA) for text classification tasks.

    https://arxiv.org/pdf/1901.11196.pdf
    https://github.com/jasonwei20/eda_nlp

    Parameters
    ------------
    max_augment : int, default=5
        Maximum number of augmentations to apply. Number of augmentation depends on the text input like how words with multiple synonym.

    synonym_replacement_prob : float, default=0.1
        Probability of synonym replacement of a random word. Slow

    synonym_insertion_prob : float, default=0.1
        Probability of synonym insertion of a random word. Slow

    random_swapping_prob : float, default=0.1
        Probability of randomly swapping placement of the two words.

    random_deletion_prob : float, default=0.1
        Probability of randomly deleting a word.
    """

    def __init__(self,
                 max_augment: int = 5,
                 synonym_replacement_prob: float = 0.1,
                 synonym_insertion_prob: float = 0.1,
                 random_swapping_prob: float = 0.1,
                 random_deletion_prob: float = 0.1):

        self.max_augment = max_augment
        self.synonym_replacement_prob = synonym_replacement_prob
        self.synonym_insertion_prob = synonym_insertion_prob
        self.random_swapping_prob = random_swapping_prob
        self.random_deletion_prob = random_deletion_prob

    def fit(self, X: pd.Series, y=None, **extra_params):
        """Does nothing. Exist for compatibility reasons for sklearn pipelines."""
        return self

    def transform(self, X: pd.Series, y=None, **extra_params):
        """Augmentate text from given text series.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : Optional[pd.Series]
            None or Pandas text series containing targets. If provided augmented target series returned.

        Returns
        -------
        X_auged : pd.Series
            Augmented text series.

        y_auged : pd.Series or None
            Augmented target series.
        """
        validate_text_input(X)

        eda_partial = functools.partial(eda,
                                        num_aug=self.max_augment,
                                        alpha_sr=self.synonym_replacement_prob,
                                        alpha_ri=self.synonym_insertion_prob,
                                        alpha_rs=self.random_swapping_prob,
                                        p_rd=self.random_deletion_prob
                                        )

        X_auged = X.progress_apply(eda_partial).apply(pd.Series).stack()

        if y is None:
            return X_auged.reset_index(drop=True)

        # EDA doesnt always yield same number of augmentations in self.max_augment.
        # Make sure y is duplicated correctly
        counts = X_auged.groupby(level=[0]).size()
        tmp = pd.concat([y, counts], axis=1)
        tmp = tmp.apply(_duplicator_pandas, axis=1).tolist()
        tmp = [item for sublist in tmp for item in sublist]

        y_auged = pd.Series(tmp)
        X_auged = X_auged.reset_index(drop=True)

        return X_auged, y_auged

    def fit_transform(self, X, y=None, **extra_params):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.Series
            Pandas text series containing texts.

        y : Optional[pd.Series]
            None or Pandas text series containing targets. If provided augmented target series returned.

        Returns
        -------
        X_auged : pd.Series
            Augmented text series.

        y_auged : pd.Series or None
            Augmented target series.
        """
        return self.fit(X, y, **extra_params).transform(X, y, **extra_params)
