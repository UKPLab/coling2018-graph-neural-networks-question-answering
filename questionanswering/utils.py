# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

# Embeddings and vocabulary utility methods
import sys

import abc
import numpy as np
import logging
import re
import nltk
import yaml

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"

special_tokens = {"&ndash;": "–",
                  "&mdash;": "—",
                  "@card@": "0"
                  }


def load(path):
    """
    Loads pre-trained embeddings from the specified path.

    @return (embeddings as an numpy array, word to index dictionary)
    """
    word2idx = {}  # Maps a word to the index in the embeddings matrix
    embeddings = []

    with open(path, 'r') as fIn:
        idx = 1
        for line in fIn:
            split = line.strip().split(' ')
            embeddings.append(np.array([float(num) for num in split[1:]]))
            word2idx[split[0]] = idx
            idx += 1

    word2idx[all_zeroes] = 0
    embedding_size = embeddings[0].shape[0]
    logger.debug("Emb. size: {}".format(embedding_size))
    embeddings = np.asarray([[0.0]*embedding_size] + embeddings, dtype='float32')

    rare_w_ids = list(range(idx-101,idx-1))
    unknown_emb = np.average(embeddings[rare_w_ids,:], axis=0)
    embeddings = np.append(embeddings, [unknown_emb], axis=0)
    word2idx[unknown_el] = idx
    idx += 1

    logger.debug("Loaded: {}".format(embeddings.shape))

    return embeddings, word2idx


def get_idx(word, word2idx):
    """
    Get the word index for the given word. Maps all numbers to 0, lowercases if necessary.

    :param word: the word in question
    :param word2idx: dictionary constructed from an embeddings file
    :return: integer index of the word
    """
    unknown_idx = word2idx[unknown_el]
    word = word.strip()
    if word in word2idx:
        return word2idx[word]
    elif word.lower() in word2idx:
        return word2idx[word.lower()]
    elif word in special_tokens:
        return word2idx[special_tokens[word]]
    trimmed = re.sub("(^\W|\W$)", "", word)
    if trimmed in word2idx:
        return word2idx[trimmed]
    elif trimmed.lower() in word2idx:
        return word2idx[trimmed.lower()]
    no_digits = re.sub("([0-9][0-9.,]*)", '0', word)
    if no_digits in word2idx:
        return word2idx[no_digits]
    return unknown_idx


def get_trigram_index(sentences):
    """
    Create a trigram index from the list of tokenized sentences.

    :param sentences: list of list of tokens
    :return: trigram to index mapping
    >>> len(get_trigram_index([['who', 'played', 'whom']]))
    11
    """
    trigram_set = {t for tokens in sentences for t in tokens_to_trigrams(tokens)}
    trigram2idx = {t: i for i, t in enumerate(trigram_set, 1)}
    trigram2idx[all_zeroes] = 0
    trigram2idx[unknown_el] = len(trigram2idx)
    return trigram2idx

def tokens_to_trigrams(tokens):
    """
    Convert a list of tokens to a list of trigrams following the hashing technique.

    :param tokens: list of tokens
    :return: list of triples of characters
    >>> tokens_to_trigrams(['who', 'played', 'bond'])
    [('#', 'w', 'h'), ('w', 'h', 'o'), ('h', 'o', '#'), ('#', 'p', 'l'), ('p', 'l', 'a'), ('l', 'a', 'y'), ('a', 'y', 'e'), ('y', 'e', 'd'), ('e', 'd', '#'), ('#', 'b', 'o'), ('b', 'o', 'n'), ('o', 'n', 'd'), ('n', 'd', '#')]
    """
    return [trigram for t in tokens for trigram in nltk.ngrams("#{}#".format(t), 3)]

def get_character_index(sentences):
    """
    Create a character index from a list of sentences. Each sentence is a string.

    :param sentences: list of strings
    :return: character to index mapping
    >>> len(get_character_index(['who played whom']))
    11
    """
    character_set = {c for sent in sentences for c in sent}
    character2idx = {c: i for i, c in enumerate(character_set, 1)}
    character2idx[all_zeroes] = 0
    character2idx[unknown_el] = len(character2idx)
    return character2idx


def get_word_index(tokens):
    """
    Create a character index from a list of sentences. Each sentence is a string.

    :param tokens: list of strings
    :return: character to index mapping
    """
    token_set = set(tokens)
    word2idx = {t: i for i, t in enumerate(token_set, 1)}
    word2idx[all_zeroes] = 0
    word2idx[unknown_el] = len(word2idx)
    return word2idx


class Loggable(metaclass=abc.ABCMeta):

    def __init__(self, logger=None, **kwargs):
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file.read())
    print(config)
    if "webquestions" not in config:
        print("Dataset location not in the config file!")
        sys.exit()
    if "model" not in config:
        print("Model params not in the config file!")
        sys.exit()
    if "wikidata" not in config:
        print("Wikidata parameters not in the config file!")
        sys.exit()
    return config


def load_blacklist(path_to_list):
    try:
        with open(path_to_list) as f:
            return_list = {l.strip() for l in f.readlines()}
        return return_list
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
    try:
        with open("../" + path_to_list) as f:
            return_list = {l.strip() for l in f.readlines()}
        return return_list
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
        return set()


def load_property_labels(path_to_property_labels):
    try:
        with open(path_to_property_labels) as infile:
            return_map = {l.split("\t")[0]: l.split("\t")[1].strip().lower() for l in infile.readlines()}
        return return_map
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
        return {}


def load_entity_map(path_to_map):
    """
    Load the map of entity labels from a file.

    :param path_to_map: location of the map file
    :return: entity map as an nltk.Index
    """
    try:
        with open(path_to_map) as f:
            return_map = [l.strip().split("\t") for l in f.readlines()]
        return nltk.Index([(t[1], t[0]) for t in return_map])
    except Exception as ex:
        logger.error("No entity map found. {}".format(ex))
    try:
        with open("../" + path_to_map) as f:
            return_map = [l.strip().split("\t") for l in f.readlines()]
        return nltk.Index([(t[1], t[0]) for t in return_map])
    except Exception as ex:
        logger.error("No entity map found. {}".format(ex))
        return {"Q76": ["Barack Obama"]}


RESOURCES_FOLDER = "../resources/"