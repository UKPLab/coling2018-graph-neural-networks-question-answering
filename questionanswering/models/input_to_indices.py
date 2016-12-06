import numpy as np
import tqdm
import utils
import nltk

unknown = ('#', '#', '#')
all_zeroes = (0,)

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


def encode_by_trigrams(graphs, trigram2idx, property2label, max_sent_len=70, max_property_len=70, verbose=False):
    graphs = [el for el in graphs if el]
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    edges_matrix = np.zeros((len(graphs), len(graphs[0]),  max_property_len), dtype="int32")

    for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
        tokens = graph_set[0].get("tokens", [])
        trigrams = tokens_to_trigrams(tokens)
        trigram_ids = [trigram2idx.get(t, trigram2idx[unknown]) for t in trigrams]
        sentences_matrix[index, :len(trigram_ids)] = trigram_ids[:max_sent_len]
        for g_index, g in enumerate(graph_set):
            if g["edgeSet"]:
                property_label = property2label.get(g["edgeSet"][0].get('kbID', '')[:-1], utils.unknown)
                property_label += " " + g["edgeSet"][0].get('type', '')
                trigrams = tokens_to_trigrams(property_label.split())
                edge_ids = [trigram2idx.get(t, trigram2idx[unknown]) for t in trigrams]
                edges_matrix[index, g_index, :len(edge_ids)] = edge_ids[:max_property_len]

    return sentences_matrix, edges_matrix


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


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
