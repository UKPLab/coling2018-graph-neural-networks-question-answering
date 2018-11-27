# Embeddings and vocabulary utility methods
import codecs
import logging
import re
import os
from collections import defaultdict

import json
import nltk
import numpy as np
from pycorenlp import StanfordCoreNLP
from typing import Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"
epsilon = 10e-8

special_tokens = {"&ndash;": "–",
                  "&mdash;": "—",
                  "@card@": "0"
                  }

corenlp = StanfordCoreNLP('http://semanticparsing:9000')
corenlp_properties = {
    'annotators': 'tokenize, pos, ner',
    'outputFormat': 'json'
}
corenlp_caseless = {
    'pos.model': 'edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger',
    'ner.model': #'edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz,' +
                 'edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz,'
                 #+ 'edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz'
}


module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)
RESOURCES_FOLDER = os.path.join(module_location, "..", "resources/")

split_pattern = re.compile(r"[\s'-:,]")


def get_tagged_from_server(input_text, caseless=False):
    """
    Get pos tagged and ner from the CoreNLP Server

    :param input_text: input text as a string
    :return: tokenized text with pos and ne tags
    >>> get_tagged_from_server("Light explodes over Pep Guardiola's head in Bernabeu press room. Will Mourinho stop at nothing?! Heh heh")[0] == \
    {'characterOffsetBegin': 0, 'ner': 'O', 'pos': 'JJ', 'characterOffsetEnd': 5, 'originalText': 'Light', 'lemma': 'light'}
    True
    """
    if len(input_text.strip()) == 0:
        return []
    if "@" in input_text or "#" in input_text:
        input_text = _preprocess_twitter_handles(input_text)
    input_text = remove_links(input_text)
    input_text = _preprocess_corenlp_input(input_text)
    if caseless:
        input_text = input_text.lower()
    corenlp_output = corenlp.annotate(input_text,
                                      properties={**corenlp_properties, **corenlp_caseless} if caseless else corenlp_properties
                                      ).get("sentences", [])
    tagged = [{k: t[k] for k in {"index", "originalText", "pos", "ner", "lemma", "characterOffsetBegin", "characterOffsetEnd"}}
              for sent in corenlp_output for t in sent['tokens']]
    return tagged


def _preprocess_corenlp_input(input_text):
    input_text = input_text.replace("/", " / ")
    input_text = input_text.replace("-", " - ")
    input_text = input_text.replace("–", " – ")
    input_text = input_text.replace("_", " _ ")
    return input_text


def remove_links(input_text):
    """
    Remove links from the input text.
    
    :param input_text: 
    :return:
    >>> remove_links('The Buccaneers just gave a $19 million contract to a punter http://t.co/ZYTqUhn/jhjhf?asas=sad via @89YahooSports wow')
    'The Buccaneers just gave a $19 million contract to a punter via @89YahooSports wow'
    >>> remove_links('The Buccaneers just gave a $19 million contract to a punter http://t.co/ZYTqUhn/jhjh')
    'The Buccaneers just gave a $19 million contract to a punter '
    >>> remove_links('The Buccaneers just gave a www.google.com was there.')
    'The Buccaneers just gave a was there.'
    >>> remove_links('The Buccaneers just gave a www.goo-gle.com was there.')
    'The Buccaneers just gave a was there.'
    """
    input_text = re.sub(r"(https?://www\.|https?://|www\.)[\w\-]+?(\.[a-zA-Z]+)+(/[^\s]+)*\s*", "", input_text)
    return input_text


def _preprocess_twitter_handles(input_text):
    """
    Split twitter handles and hashtags into tokens when possible
    
    :param input_text: 
    :return:
    >>> _preprocess_twitter_handles('The Buccaneers just gave a $19 million contract to a punter via @89YahooSports wow')
    'The Buccaneers just gave a $19 million contract to a punter via @89 Yahoo Sports wow'
    >>> _preprocess_twitter_handles('Congrats to my first ever Broski of the Week @CMPunk!')
    'Congrats to my first ever Broski of the Week @CM Punk!'
    """
    input_text = re.sub(r'([\w]+?)([A-Z])(?=[a-z0-9])', r'\1 \2', input_text)
    return input_text  #re.sub(r"[@#](?=[\w0-9])", "", input_text)


def _tagged2tuples(tagged_dicts):
    """
    >>> _tagged2tuples(get_tagged_from_server("who has starred in the movie die hard?"))
    [('who', 'O', 'WP'), ('has', 'O', 'VBZ'), ('starred', 'O', 'VBN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('movie', 'O', 'NN'), ('die', 'O', 'VB'), ('hard', 'O', 'RB'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("What was the last queen album?"))
    [('What', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('last', 'O', 'JJ'), ('queen', 'O', 'NN'), ('album', 'O', 'NN'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("What was the first queen album?"))
    [('What', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('first', 'ORDINAL', 'JJ'), ('queen', 'O', 'NN'), ('album', 'O', 'NN'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("What actors star in the Big Bang Theory?"))
    [('What', 'O', 'WDT'), ('actors', 'O', 'NNS'), ('star', 'O', 'NN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('Big', 'O', 'NNP'), ('Bang', 'O', 'NNP'), ('Theory', 'O', 'NNP'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("what actors star in the big bang theory?", caseless=True))
    [('what', 'O', 'WDT'), ('actors', 'O', 'NNS'), ('star', 'O', 'NN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('big', 'O', 'JJ'), ('bang', 'O', 'NN'), ('theory', 'O', 'NN'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("who wrote the song hotel california?", caseless=True))
    [('who', 'O', 'WP'), ('wrote', 'O', 'VBD'), ('the', 'O', 'DT'), ('song', 'O', 'NN'), ('hotel', 'O', 'NN'), ('california', 'LOCATION', 'NNP'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("who was the president of the united states in 2012?", caseless=True))
    [('who', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('president', 'O', 'NN'), ('of', 'O', 'IN'), ('the', 'O', 'DT'), ('united', 'LOCATION', 'NNP'), ('states', 'LOCATION', 'NNPS'), ('in', 'O', 'IN'), ('2012', 'DATE', 'CD'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("who plays megan in the movie taken?", caseless=True))
    [('who', 'O', 'WP'), ('plays', 'O', 'VBZ'), ('megan', 'O', 'NNP'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('movie', 'O', 'NN'), ('taken', 'O', 'VBN'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("what language do canadians speak?", caseless=True))
    [('what', 'O', 'WDT'), ('language', 'O', 'NN'), ('do', 'O', 'VBP'), ('canadians', 'O', 'NNPS'), ('speak', 'O', 'VB'), ('?', 'O', '.')]
    >>> _tagged2tuples(get_tagged_from_server("Light explodes over Pep Guardiola's head in Bernabeu press room. Will Mourinho stop at nothing?! Heh heh"))
    [('Light', 'O', 'JJ'), ('explodes', 'O', 'VBZ'), ('over', 'O', 'IN'), ('Pep', 'PERSON', 'NNP'), ('Guardiola', 'PERSON', 'NNP'), ("'s", 'O', 'POS'), ('head', 'O', 'NN'), ('in', 'O', 'IN'), ('Bernabeu', 'LOCATION', 'NNP'), ('press', 'O', 'NN'), ('room', 'O', 'NN'), ('.', 'O', '.'), ('Will', 'O', 'MD'), ('Mourinho', 'PERSON', 'NNP'), ('stop', 'O', 'VB'), ('at', 'O', 'IN'), ('nothing', 'O', 'NN'), ('?!', 'NUMBER', 'CD'), ('Heh', 'O', 'NNP'), ('heh', 'O', 'RB')]
    """
    tagged = [(t['originalText'], t['ner'], t['pos']) for t in tagged_dicts]
    return tagged


def _lemmatize_tokens(entity_tokens):
    """
    Lemmatize the list of tokens using the Stanford CoreNLP.

    :param entity_tokens:
    :return:
    >>> _lemmatize_tokens(['House', 'Of', 'Representatives'])
    ['House', 'Of', 'Representative']
    >>> _lemmatize_tokens(['Canadians'])
    ['Canadian']
    >>> _lemmatize_tokens(['star', 'wars'])
    ['star', 'war']
    >>> _lemmatize_tokens(['Movie', 'does'])
    ['Movie', 'do']
    >>> _lemmatize_tokens("who is the member of the house of representatives?".split())
    ['who', 'be', 'the', 'member', 'of', 'the', 'house', 'of', 'representative', '?']
    """
    try:
        lemmas = corenlp.annotate(" ".join([t.lower() for t in entity_tokens]), properties={
            'annotators': 'tokenize, lemma',
            'outputFormat': 'json'
        }).get("sentences", [])[0]['tokens']
    except:
        lemmas = []
    lemmas = [t['lemma'] for t in lemmas]
    lemmas = [l.title() if i < len(entity_tokens) and entity_tokens[i].istitle() else l for i, l in enumerate(lemmas)]
    return lemmas


def load_resource_file_backoff(f):
    def load_method(file_name):
        try:
            return f(file_name)
        except Exception as ex:
            logger.error("No file found. {}".format(ex))
        try:
            return f("../" + file_name)
        except Exception as ex:
            logger.error("No file  found. {}".format(ex))
        return None
    return load_method


def load_word_embeddings(path):
    """
    Loads pre-trained embeddings from the specified path.

    @return (embeddings as an numpy array, word to index dictionary)
    """
    word2idx = defaultdict(lambda: 1)  # Maps a word to the index in the embeddings matrix
    word2idx[all_zeroes] = 0
    word2idx[unknown_el] = 1
    embeddings = []

    with codecs.open(path, 'r', encoding='utf-8') as fIn:
        idx = 2
        for line in fIn:
            split = line.strip().split(' ')
            embeddings.append([float(num) for num in split[1:]])
            word2idx[split[0]] = idx
            idx += 1

    embedding_size = len(embeddings[0])
    embeddings = np.asarray(embeddings, dtype='float32')

    unknown_emb = np.average(embeddings[:10000], axis=0)
    embeddings = np.concatenate((np.zeros((1, embedding_size)),
                                 np.expand_dims(unknown_emb, 0),
                                 embeddings), axis=0)
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
        return word2idx.get(special_tokens[word], unknown_idx)
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


def get_elements_index(element_set: Set):
    """
    Create an element to index mapping, that includes a zero and an unknown element.

    :param element_set: set of elements to enumerate
    :return: an index as a dictionary
    >>> get_elements_index({"a", "b", "c", all_zeroes})["_UNKNOWN"] 
    4
    """
    element_set = element_set - {all_zeroes, unknown_el}
    element_set = sorted(list(element_set))
    el2idx = {c: i for i, c in enumerate(element_set, 1)}
    el2idx[all_zeroes] = 0
    el2idx[unknown_el] = len(el2idx)
    return el2idx


@load_resource_file_backoff
def load_json_resource(path_to_file):
    with open(path_to_file) as f:
        resource = json.load(f)
    return resource


@load_resource_file_backoff
def load_property_labels(path_to_property_labels):
    """
    
    :param path_to_property_labels: 
    :return:
    >>> load_property_labels("../resources/properties_with_labels.txt")["P106"]
    {'type': 'wikibase-item', 'altlabel': ['employment', 'craft', 'profession', 'job', 'work', 'career'], 'freq': 2290043, 'label': 'occupation'}
    """
    with open(path_to_property_labels) as infile:
        return_map = {}
        for l in infile.readlines():
            if not l.startswith("#"):
                columns = l.split("\t")
                return_map[columns[0].strip()] = {"label": columns[1].strip().lower(),
                                                  "altlabel": list(set(columns[3].strip().lower().split(", "))),
                                                  "type":  columns[4].strip().lower(),
                                                  "freq": int(columns[5].strip().replace(",",""))}
    return return_map


@load_resource_file_backoff
def load_entity_freq_map(path_to_map):
    """
    Load the map of entity frequencies from a file.

    :param path_to_map: location of the map file
    :return: entity map as a dictionary
    >>> load_entity_freq_map("../resources/wikidata_entity_freqs.map")['Q76']
    7070
    """
    with open(path_to_map) as f:
        return_map = [tuple(l.strip().split("\t")) for l in f.readlines()]
        return_map = [(k, int(v)) for k, v in return_map]
    return dict(return_map)


@load_resource_file_backoff
def load_entity_map(path_to_map):
    """
    Load the map of entity labels from a file.

    :param path_to_map: location of the map file
    :return: entity map as an nltk.Index
    """
    with open(path_to_map) as f:
        return_map = [l.strip().split("\t") for l in f.readlines()]
    return nltk.Index([(t[1], (t[0], t[2])) for t in return_map])


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


corenlp_pos_tagset = load_blacklist(RESOURCES_FOLDER + "/PENN.pos.tagset")


def map_pos(pos):
    if pos.endswith("S") or pos.endswith("R"):
        return pos[:-1]
    return pos
