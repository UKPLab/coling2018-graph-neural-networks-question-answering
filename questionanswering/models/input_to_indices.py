import numpy as np
import tqdm
import utils
import nltk


def encode_by_tokens(graphs, max_sent_len, max_property_len, word2idx, property2label, verbose=False):
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    edges_matrix = np.zeros((len(graphs), max_property_len), dtype="int32")

    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        token_ids = [utils.get_idx(t, word2idx) for t in  g.get("tokens", [])][:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        if g["edgeSet"]:
            property_label = property2label.get(g["edgeSet"][0]['kbID'][:-1], utils.unknown)
            edge_ids = [utils.get_idx(t, word2idx) for t in property_label.split()][:max_property_len]
            edges_matrix[index, :len(edge_ids)] = edge_ids

    return sentences_matrix, edges_matrix


def encode_by_trigrams(graphs, trigram2idx, property2label, max_sent_len=36, max_property_len=20, verbose=False):
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    edges_matrix = np.zeros((len(graphs), max_property_len), dtype="int32")

    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        tokens = g.get("tokens", [])
        trigrams = tokens_to_trigrams(tokens)
        trigram_ids = [trigram2idx.get(t, 0) for t in trigrams]
        sentences_matrix[index, :len(trigram_ids)] = trigram_ids[:max_sent_len]
        if g["edgeSet"]:
            property_label = property2label.get(g["edgeSet"][0].get('kbID', '')[:-1], utils.unknown)
            edge_ids = tokens_to_trigrams(property_label.split())
            edges_matrix[index, :len(edge_ids)] = edge_ids[:max_property_len]

    return sentences_matrix, edges_matrix, trigram2idx


def tokens_to_trigrams(tokens):
    """
    Convert a list of tokens to a list of trigrams following the hashing technique.

    :param tokens: list of tokens
    :return: list of triples of characters
    >>> tokens_to_trigrams(['who', 'played', 'bond'])
    [('#', 'w', 'h'), ('w', 'h', 'o'), ('h', 'o', '#'), ('#', 'p', 'l'), ('p', 'l', 'a'), ('l', 'a', 'y'), ('a', 'y', 'e'), ('y', 'e', 'd'), ('e', 'd', '#'), ('#', 'b', 'o'), ('b', 'o', 'n'), ('o', 'n', 'd'), ('n', 'd', '#')]
    """
    return [trigram for t in tokens for trigram in nltk.ngrams("#{}#".format(t), 3)]


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
