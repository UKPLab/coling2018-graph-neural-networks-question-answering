import numpy as np
import tqdm
import utils
import nltk
import re

from construction import graph

unknown = ('#', '#', '#')
all_zeroes = (0,)
zero_character = '_zero'
unknown_character = '_unknown'


def encode_by_tokens(graphs, max_sent_len, max_property_len, word2idx, property2label, verbose=False):
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    edges_matrix = np.zeros((len(graphs), max_property_len), dtype="int32")

    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        token_ids = [utils.get_idx(t, word2idx) for t in g.get("tokens", [])][:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        if g["edgeSet"]:
            property_label = property2label.get(g["edgeSet"][0]['kbID'][:-1], utils.unknown_word)
            edge_ids = [utils.get_idx(t, word2idx) for t in property_label.split()][:max_property_len]
            edges_matrix[index, :len(edge_ids)] = edge_ids

    return sentences_matrix, edges_matrix


def encode_by_trigram(graph_set, trigram_vocabulary, property2label):
    sentence_tokens = graph_set[0].get("tokens", [])
    sentence_trigrams = [set(string_to_trigrams(token)) for token in sentence_tokens]
    sentence_encoded = [[int(t in trigrams) for t in trigram_vocabulary]
                        for trigrams in sentence_trigrams]
    edges_encoded = []
    for g in graph_set:
        first_edge = graph.get_graph_first_edge(g)
        property_label = property2label.get(first_edge.get('kbID', '')[:-1], utils.unknown_word)
        e_type = first_edge.get('type', 'direct')
        edge_trigrams = [set(string_to_trigrams(token)) for token in property_label.split()]
        edge_encoded = [[int(t in trigrams) for t in trigram_vocabulary]
                            for trigrams in edge_trigrams]
        edges_encoded.append(edge_encoded)
    return sentence_encoded, edges_encoded


def encode_batch_by_trigrams(graphs, trigram_vocabulary, property2label, max_input_len=11, verbose=False):
    graphs = [el for el in graphs if el]
    sentences_matrix = np.zeros((len(graphs), max_input_len, len(trigram_vocabulary)), dtype="int32")
    edges_matrix = np.zeros((len(graphs), len(graphs[0]),  max_input_len, len(trigram_vocabulary)), dtype="int32")

    for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        sentence_encoded, edges_encoded = encode_by_trigram(graph_set, trigram_vocabulary, property2label)
        assert len(edges_encoded) == edges_matrix.shape[1]
        sentence_encoded = sentence_encoded[:max_input_len]
        sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
        for i, edge_encoded in enumerate(edges_encoded):
            edge_encoded = edge_encoded[:max_input_len]
            edges_matrix[index, i, :len(edge_encoded)] = edge_encoded

    return sentences_matrix, edges_matrix


def encode_batch_by_character(graphs, character2idx, property2label, max_input_len=70, edge_with_entity=False, verbose=False):
    sentences_matrix = np.zeros((len(graphs), max_input_len), dtype="int32")
    edges_matrix = np.zeros((len(graphs), len(graphs[0]),  max_input_len), dtype="int32")
    for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        sentence_ids, edges_ids = encode_by_character(graph_set, character2idx, property2label, edge_with_entity)
        assert len(edges_ids) == edges_matrix.shape[1]
        sentence_ids = sentence_ids[:max_input_len]
        sentences_matrix[index, :len(sentence_ids)] = sentence_ids
        # edges_ids = [edges_ids] # What was that??
        for i, edge_ids in enumerate(edges_ids):
            edge_ids = edge_ids[:max_input_len]
            edges_matrix[index, i, :len(edge_ids)] = edge_ids
    return sentences_matrix, edges_matrix


def encode_by_character(graph_set, character2idx, property2label, edge_with_entity=False):
    sentence_str = " ".join(graph_set[0].get("tokens", []))
    sentence_ids = string_to_unigrams(sentence_str, character2idx)
    edges_ids = []
    for g in graph_set:
        first_edge = graph.get_graph_first_edge(g)
        property_label = property2label.get(first_edge.get('kbID', '')[:-1], utils.unknown_word)
        e_type = first_edge.get('type', 'direct')
        if edge_with_entity:
            if e_type == 'direct':
                property_label = property_label + " " + " ".join(first_edge.get('right', []))
            else:
                property_label = " ".join(first_edge.get('right', [])) + " " + property_label
        edges_ids.append(string_to_unigrams(property_label, character2idx))
    return sentence_ids, edges_ids


def normalize_string(input_string):
    input_string = input_string.lower()
    input_string = re.sub(r"\d+", "0", input_string)
    return input_string


def string_to_unigrams(input_string, character2idx):
    input_string = normalize_string(input_string)
    return [character2idx.get(c, character2idx[unknown_character]) for c in input_string]


def string_to_trigrams(t):
    """
    Convert a token to a list of trigrams following the hashing technique.

    :param t: a single token as a string
    :return: list of triples of characters
    >>> string_to_trigrams('who')
    [('#', 'w', 'h'), ('w', 'h', 'o'), ('h', 'o', '#')]
    """
    t = normalize_string(t)
    return nltk.ngrams("#{}#".format(t), 3)


def tokens_to_trigrams(tokens):
    """
    Convert a list of tokens to a list of trigrams following the hashing technique.

    :param tokens: list of tokens
    :return: list of triples of characters
    >>> tokens_to_trigrams(['who', 'played', 'bond'])
    [('#', 'w', 'h'), ('w', 'h', 'o'), ('h', 'o', '#'), ('#', 'p', 'l'), ('p', 'l', 'a'), ('l', 'a', 'y'), ('a', 'y', 'e'), ('y', 'e', 'd'), ('e', 'd', '#'), ('#', 'b', 'o'), ('b', 'o', 'n'), ('o', 'n', 'd'), ('n', 'd', '#')]
    """
    return [trigram for t in tokens for trigram in nltk.ngrams("#{}#".format(t), 3)]


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
    trigram2idx[unknown] = len(trigram2idx)
    return trigram2idx


def get_character_index(sentences):
    """
    Create a character index from a list of sentences. Each sentence is a string.

    :param sentences: list of strings
    :return: character to index mapping
    >>> len(get_character_index(['who played whom']))
    11
    """
    character_set = {c for sent in sentences for c in normalize_string(sent)}
    character2idx = {c: i for i, c in enumerate(character_set, 1)}
    character2idx[zero_character] = 0
    character2idx[unknown_character] = len(character2idx)
    return character2idx


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
