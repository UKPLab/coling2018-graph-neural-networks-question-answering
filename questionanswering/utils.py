# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

# Embeddings and vocabulary utility methods

import numpy as np
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown_word = "_UNKNOWN"

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
    word2idx[unknown_word] = idx
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
    unknown_idx = word2idx[unknown_word]
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