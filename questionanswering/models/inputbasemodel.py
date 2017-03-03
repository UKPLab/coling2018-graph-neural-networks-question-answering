from collections import defaultdict

import abc
import json
import keras
import nltk
import numpy as np
import tqdm

import utils
from construction import graph
from models.qamodel import TrainableQAModel
from wikidata import wdaccess


PROPERTY_VOCABULARY = {"<e>", "<num>", "<argmax>", "<argmin>", "<x>", "<v>", "<filter>"}


class CharBasedModel(TrainableQAModel, metaclass=abc.ABCMeta):
    """
    A model that encodes input on a character level, using single characters. Input is encoded as a sequence
    of indices from a vocabulary.
    """
    def __init__(self, **kwargs):
        self._character2idx = None  # This is actually ngram2idx
        super(CharBasedModel, self).__init__(**kwargs)

    def prepare_model(self, train_tokens, properties_set):
        assert "char.model" in self._p
        if self._p.get("mark.sent.boundaries", False):
            train_tokens.update({"<S>", "<E>"})
        if self._p.get('vocabulary.with.edgelabels', True):
            train_tokens.update(get_property_token_set(properties_set))
        self._character2idx = utils.get_elements_index({c for token in train_tokens for c in string_to_ngrams(token, self._p["char.model"].get("ngram.len", 1))})
        self.logger.debug('Character index created, size: {}'.format(len(self._character2idx)))
        with open(self._save_model_to + "character2idx_{}.json".format(self._model_number), 'w') as out:
            json.dump(self._character2idx, out, indent=2)
        self._p['vocab.size'] = len(self._character2idx)

        super(CharBasedModel, self).prepare_model(train_tokens, properties_set)

    def _encode_string(self, s):
        return [self._character2idx.get(c, self._character2idx[utils.unknown_el]) for c in string_to_ngrams(s, self._p["char.model"].get("ngram.len", 1))]

    def encode_question(self, instance):
        sentence_tokens, graph_set = instance
        sentence_tokens = self._preprocess_sentence_tokens(sentence_tokens, graph_set)
        sentence_str = " ".join(sentence_tokens)
        sentence_encoded = self._encode_string(sentence_str)
        return sentence_encoded

    def encode_graphs(self, instance):
        _, graph_set = instance
        graphs_encoded = []
        for g in graph_set:
            edges_encoded = []
            for edge in g.get('edgeSet', []):
                property_label = self._get_edge_label(edge)
                edges_encoded.append(self._encode_string(property_label))
            graphs_encoded.append(edges_encoded)
        return graphs_encoded

    @abc.abstractmethod
    def encode_data_instance(self, instance, **kwargs):
        sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
        sentence_ids = keras.preprocessing.sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int16")
        graph_matrix = np.zeros((len(graphs_encoded), self._p.get('max.graph.size', 3), self._p.get('max.sent.len', 70)), dtype="int16")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 70)]
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_batch(self, graphs, verbose=False):
        graphs = [el for el in graphs if len(el) == 2 and len(el[1]) > 0]
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 70)), dtype="int16")
        graph_matrix = np.zeros((len(graphs), len(graphs[0][1]), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 70)), dtype="int16")
        for index, instance in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 70)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_encoded in enumerate(graph_encoded):
                    edge_encoded = edge_encoded[:self._p.get('max.sent.len', 70)]
                    graph_matrix[index, i, j, :len(edge_encoded)] = edge_encoded
        return sentences_matrix, graph_matrix

    def load_from_file(self, path_to_model):
        super(CharBasedModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: character2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "character2idx_{}.json".format(self._model_number)) as f:
            self._character2idx = json.load(f)
        self._p['vocab.size'] = len(self._character2idx)
        self.logger.debug("Vocabulary size: {}.".format(len(self._character2idx)))


class TrigramBasedModel(TrainableQAModel, metaclass=abc.ABCMeta):
    """
    A model that uses the hashing trick to encode the input. Each word is represented as a vector of
    of boolean values, the dimensionality is the size of the trigram vocabulary.
    """

    def __init__(self, **kwargs):
        self._trigram_vocabulary = None
        super(TrigramBasedModel, self).__init__(**kwargs)

    def prepare_model(self, train_tokens, properties_set):
        if self._p.get("mark.sent.boundaries", False):
            train_tokens.update({"<S>", "<E>"})
        if self._p.get('vocabulary.with.edgelabels', True):
            train_tokens.update(get_property_token_set(properties_set))
        self._trigram_vocabulary = list({t for token in train_tokens for t in string_to_trigrams(token)})
        self.logger.debug('Trigram vocabulary created, size: {}'.format(len(self._trigram_vocabulary)))
        if len(self._trigram_vocabulary) > 0:
            self._p['vocab.size'] = len(self._trigram_vocabulary)
            with open(self._save_model_to + "trigram_vocabulary_{}.json".format(self._model_number), 'w') as out:
                json.dump(self._trigram_vocabulary, out, indent=2)
        super(TrigramBasedModel, self).prepare_model(train_tokens, properties_set)

    def _encode_tokens(self, tokens):
        sentence_trigrams = [set(string_to_trigrams(token)) for token in tokens]
        return [[int(t in trigrams) for t in self._trigram_vocabulary] for trigrams in sentence_trigrams]

    def encode_question(self, instance):
        sentence_tokens, graph_set = instance
        sentence_tokens = self._preprocess_sentence_tokens(sentence_tokens, graph_set)
        sentence_encoded = self._encode_tokens(sentence_tokens)
        return sentence_encoded

    def encode_graphs(self, instance):
        _, graph_set = instance
        graphs_encoded = []
        for g in graph_set:
            edges_encoded = []
            for edge in g.get('edgeSet', []):
                property_label = graph.get_property_str_representation(edge, wdaccess.property2label, self._p.get("replace.entities", False))
                if self._p.get("mark.sent.boundaries", False):
                    property_label = "<S> " + property_label + " <E>"
                edge_encoded = self._encode_tokens(property_label.split())
                edges_encoded.append(edge_encoded)
            graphs_encoded.append(edges_encoded)
        return graphs_encoded

    @abc.abstractmethod
    def encode_data_instance(self, instance):
        sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
        sentence_ids = keras.preprocessing.sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int8")
        graph_matrix = np.zeros((len(graphs_encoded), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_batch(self, graphs, verbose=False):
        graphs = [el for el in graphs if len(el) == 2 and len(el[1]) > 0]
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        graph_matrix = np.zeros((len(graphs), len(graphs[0][1]), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        for index, instance in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_encoded in enumerate(graph_encoded):
                    edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                    graph_matrix[index, i, j, :len(edge_encoded)] = edge_encoded
        return sentences_matrix, graph_matrix

    def load_from_file(self, path_to_model):
        super(TrigramBasedModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: trigram_vocabulary_{}.json".format(self._model_number))
        with open(self._save_model_to + "trigram_vocabulary_{}.json".format(self._model_number)) as f:
            self._trigram_vocabulary = json.load(f)
        self._trigram_vocabulary = [t for t in self._trigram_vocabulary]
        self._p['vocab.size'] = len(self._trigram_vocabulary)
        self.logger.debug("Vocabulary size: {}.".format(len(self._trigram_vocabulary)))


class WordBasedModel(TrainableQAModel, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._embedding_matrix = None
        self._word2idx = defaultdict(int)
        super(WordBasedModel, self).__init__(**kwargs)

    def prepare_model(self, train_tokens, properties_set):
        if "word.embeddings" in self._p:
            self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
            self.logger.debug('Word index loaded, size: {}'.format(len(self._word2idx)))
        else:
            if self._p.get("mark.sent.boundaries", False):
                train_tokens.update({"<S>", "<E>"})
            if self._p.get('vocabulary.with.edgelabels', True):
                train_tokens.update(get_property_token_set(properties_set))
            self._word2idx = utils.get_word_index([t for tokens in train_tokens for t in tokens])
            self.logger.debug('Word index created, size: {}'.format(len(self._word2idx)))
            with open(self._save_model_to + "word2idx_{}.json".format(self._model_number), 'w') as out:
                json.dump(self._word2idx, out, indent=2)
        super(WordBasedModel, self).prepare_model(train_tokens, properties_set)

    def encode_question(self, graph_set):
        sentence_tokens = graph_set[0].get("tokens", [])
        sentence_encoded = [utils.get_idx(t, self._word2idx) for t in sentence_tokens]
        return sentence_encoded

    def encode_graphs(self, graph_set):
        graphs_encoded = []
        for g in graph_set:
            edges_encoded = []
            for edge in g.get('edgeSet', []):
                property_label = edge.get('label', '')
                edge_ids = [utils.get_idx(t, self._word2idx) for t in property_label.split()]
                edges_encoded.append(edge_ids)
            graphs_encoded.append(edges_encoded)
        return graphs_encoded

    @abc.abstractmethod
    def encode_data_instance(self, instance):
        sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
        sentence_ids = keras.preprocessing.sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graph_matrix = np.zeros((len(graphs_encoded), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10)), dtype="int8")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_batch(self, graphs, verbose=False):
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10)), dtype="int32")
        graph_matrix = np.zeros((len(graphs), len(graphs[0]), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10)), dtype="int8")

        for index, instance in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, graphs_encoded = self.encode_question(instance), self.encode_graphs(instance)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_encoded in enumerate(graph_encoded):
                    edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                    graph_matrix[index, i, j, :len(edge_encoded)] = edge_encoded
        return sentences_matrix, graph_matrix

    def load_from_file(self, path_to_model):
        super(WordBasedModel, self).load_from_file(path_to_model=path_to_model)

        if "word.embeddings" in self._p:
            self.logger.debug("Loading pre-trained word embeddings.")
            self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
        else:
            self.logger.debug("Loading vocabulary from: word2idx_{}.json".format(self._model_number))
            with open(self._save_model_to + "word2idx_{}.json".format(self._model_number)) as f:
                self._word2idx = json.load(f)
        self.logger.debug("word2idx size: {}.".format(len(self._word2idx)))


class GraphFeaturesModel(WordBasedModel, TrainableQAModel, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._property2idx = {utils.all_zeroes: 0, utils.unknown_el: 1}
        self._propertytype2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "v": 2, "q": 3}
        self._type2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "direct": 2, "reverse": 3, "v-structure": 4, "time": 5}
        self._modifier2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "argmax": 2, "argmin": 3, "num": 4, "filter": 5}
        self._feature_vector_size = sum(v if type(v) == int else 1 for f, v in self._p.get('symbolic.features', {}).items())
        self.logger.debug("Feature vector size: {}".format(self._feature_vector_size))
        super(GraphFeaturesModel, self).__init__(**kwargs)

    def prepare_model(self, train_tokens, properties_set):
        assert len(properties_set) > 0
        self.init_property_index(properties_set)
        super(GraphFeaturesModel, self).prepare_model(train_tokens, properties_set)

    def init_property_index(self, properties_set):
        properties_set = properties_set | wdaccess.HOP_UP_RELATIONS | wdaccess.HOP_DOWN_RELATIONS
        self._property2idx.update({p: i for i, p in enumerate(properties_set, start=len(self._property2idx))})
        self.logger.debug("Property index is finished: {}".format(len(self._property2idx)))
        with open(self._save_model_to + "property2idx_{}.json".format(self._model_number), 'w') as out:
            json.dump(self._property2idx, out, indent=2)

    @abc.abstractmethod
    def encode_data_instance(self, instance):
        sentence_encoded = self.encode_question(instance[:1])
        sentence_ids = keras.preprocessing.sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graphs_encoded = self.encode_graphs(instance)
        graph_matrix = np.zeros((len(instance), self._p.get('max.graph.size', 3), self._feature_vector_size), dtype="int32")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_graphs(self, graph_set):
        graphs_encoded = []
        for i, g in enumerate(graph_set):
            graph_encoded = []
            for j, edge in enumerate(g.get("edgeSet", [])[:self._p.get('max.graph.size', 3)]):
                edge_feature_vector = self.get_edge_feature_vector(edge)
                graph_encoded.append(edge_feature_vector)
            graphs_encoded.append(graph_encoded)
        return graphs_encoded

    def get_edge_feature_vector(self, edge):
        edge_kbid = edge.get('kbID')[:-1] if 'kbID' in edge else utils.unknown_el
        right_label_ids = [utils.get_idx(t, self._word2idx) for t in edge.get('canonical_right', "").split()][
                          :self._p.get('symbolic.features', {}).get("right.label", 0)]
        feature_vector = [self._property2idx.get(edge_kbid, 0),
                          self._property2idx.get(edge['hopUp'][:-1] if 'hopUp' in edge else utils.all_zeroes, 0),
                          self._property2idx.get(edge['hopDown'][:-1] if 'hopDown' in edge else utils.all_zeroes, 0),
                          self._modifier2idx.get("argmax" if "argmax" in edge
                                                 else "argmin" if "argmin" in edge
                          else "num" if "num" in edge
                          else "filter" if "filter" in edge
                          else utils.all_zeroes, 0), self._type2idx.get(edge.get('type', utils.unknown_el), 0),
                          self._propertytype2idx.get(edge['kbID'][-1] if 'kbID' in edge else utils.unknown_el, 0),
                          ] + right_label_ids
        assert len(feature_vector) <= self._feature_vector_size
        return feature_vector

    @abc.abstractmethod
    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))

        sentences_matrix = keras.preprocessing.sequence.pad_sequences([self.encode_question(i[:1]) for i in input_set],
                                                                  maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graph_matrix = np.zeros((len(input_set), len(input_set[0]), self._p.get('max.graph.size', 3), self._feature_vector_size), dtype="int32")
        for s in range(len(input_set)):
            graphs_encoded = self.encode_graphs(s)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_vector in enumerate(graph_encoded):
                    graph_matrix[s, i, j, :len(edge_vector)] = edge_vector
        return sentences_matrix, graph_matrix, targets

    def load_from_file(self, path_to_model):
        super(GraphFeaturesModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading property index from: property2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "property2idx_{}.json".format(self._model_number)) as f:
            self._property2idx = json.load(f)
        self.logger.debug("Property2idx size: {}.".format(len(self._property2idx)))


def get_property_token_set(properties_set):
    token_set = set()
    token_set.update(PROPERTY_VOCABULARY)
    token_set.update(wdaccess.HOP_DOWN_RELATIONS)
    token_set.update(wdaccess.HOP_UP_RELATIONS)
    token_set.update(wdaccess.TEMPORAL_RELATIONS)
    token_set.update({w for p in properties_set for w in wdaccess.property2label.get(p, utils.unknown_el).split()})
    return token_set


def string_to_unigrams(input_string, character2idx):
    """
    Depricated

    :param input_string:
    :param character2idx:
    :return:
    """
    return [character2idx.get(c, character2idx[utils.unknown_el]) for c in input_string]


def string_to_ngrams(t, n=1):
    """
    Convert a token to a list of ngrams. Word boundaries are marked with hashes

    :param t: a single token as a string
    :param n: ngram size
    :return: list of triples of characters
    >>> list(string_to_ngrams('who', n=3))
    ['#wh', 'who', 'ho#']
    >>> list(string_to_ngrams('who'))
    ['#', 'w', 'h', 'o', '#']
    """
    assert n > 0
    return ["".join(ngram) for ngram in nltk.ngrams("#{}#".format(t), n)]


def string_to_trigrams(t):
    """
    Convert a token to a list of trigrams.

    :param t: a single token as a string
    :return: list of triples of characters
    >>> list(string_to_trigrams('who'))
    ['#wh', 'who', 'ho#']
    """
    return string_to_ngrams(t, n=3)
