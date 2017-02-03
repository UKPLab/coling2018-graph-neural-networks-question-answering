import json
from collections import defaultdict

import keras
import nltk
import numpy as np
import tqdm
from keras.preprocessing import sequence
from keras import backend as K

import utils
from construction import graph
from models import keras_extensions
from models.word_based import WordCNNModel
from models.qamodel import TwinsModel, BrothersModel
from wikidata import wdaccess


class CharCNNModel(TwinsModel):
    def __init__(self, **kwargs):
        self._character2idx = defaultdict(int)
        super(CharCNNModel, self).__init__(**kwargs)

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sibling_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._sibling_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))

        return np.argsort(predictions)[::-1]

    def encode_data_instance(self, instance, **kwargs):
        sentence_ids, edges_ids = self.encode_by_character(instance)
        sentence_ids = sequence.pad_sequences([sentence_ids], maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int16")
        edges_ids = sequence.pad_sequences(edges_ids, maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int16")
        return sentence_ids, edges_ids

    def prepare_model(self, train_tokens, properties_set):
        if not self._character2idx:
            self._character2idx = utils.get_character_index([" ".join(tokens) for tokens in train_tokens])
            self.logger.debug('Character index created, size: {}'.format(len(self._character2idx)))
            with open(self._save_model_to + "character2idx_{}.json".format(self._model_number), 'w') as out:
                json.dump(self._character2idx, out, indent=2)
        self._p['vocab.size'] = len(self._character2idx)

        super(CharCNNModel, self).prepare_model(train_tokens, properties_set)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        characters_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int16', name='sentence_input')
        character_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=self._p['vocab.size'],
                                                      input_length=self._p['max.sent.len'],
                                                      init=self._p.get("emb.weight.init", 'uniform'),
                                                      mask_zero=False)(characters_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(character_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[characters_input], output=[semantic_vector], name=self._sibling_model_name)
        self.logger.debug("Sibling model is finished.")

        # Twins model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int16', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int16',
                                        name='edge_input')

        sentence_vector = sibiling_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(sibiling_model)(edge_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, edge_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = self.encode_batch_by_character(input_set, verbose=False)
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets

    def encode_batch_by_character(self, graphs, verbose=False):
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 70)), dtype="int16")
        edges_matrix = np.zeros((len(graphs), len(graphs[0]),  self._p.get('max.sent.len', 70)), dtype="int16")
        for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_ids, edges_ids = self.encode_by_character(graph_set)
            assert len(edges_ids) == edges_matrix.shape[1]
            sentence_ids = sentence_ids[:self._p.get('max.sent.len', 70)]
            sentences_matrix[index, :len(sentence_ids)] = sentence_ids
            # edges_ids = [edges_ids] # What was that??
            for i, edge_ids in enumerate(edges_ids):
                edge_ids = edge_ids[:self._p.get('max.sent.len', 70)]
                edges_matrix[index, i, :len(edge_ids)] = edge_ids
        return sentences_matrix, edges_matrix

    def encode_by_character(self, graph_set):
        sentence_str = " ".join(graph_set[0].get("tokens", []))
        sentence_ids = string_to_unigrams(sentence_str, self._character2idx)
        edges_ids = []
        for g in graph_set:
            first_edge = graph.get_graph_first_edge(g)
            property_label = first_edge.get('label', '')
            edges_ids.append(string_to_unigrams(property_label, self._character2idx))
        return sentence_ids, edges_ids

    def load_from_file(self, path_to_model):
        super(CharCNNModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: character2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "character2idx_{}.json".format(self._model_number)) as f:
            self._character2idx = json.load(f)
        self._p['vocab.size'] = len(self._character2idx)
        self.logger.debug("Vocabulary size: {}.".format(len(self._character2idx)))


class YihModel(TwinsModel):

    def __init__(self, **kwargs):
        self._trigram_vocabulary = []
        super(YihModel, self).__init__(**kwargs)

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = self.encode_batch_by_trigrams(input_set, verbose=False)
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets

    def prepare_model(self, train_tokens, properties_set):
        self.extract_vocabulary(train_tokens)
        super(YihModel, self).prepare_model(train_tokens, properties_set)

    def extract_vocabulary(self, train_tokens):
        if not self._trigram_vocabulary:
            self._trigram_vocabulary = list({t for tokens in train_tokens
                                             for token in tokens
                                             for t in string_to_trigrams(token)})
            self.logger.debug('Trigram vocabulary created, size: {}'.format(len(self._trigram_vocabulary)))
            with open(self._save_model_to + "trigram_vocabulary_{}.json".format(self._model_number), 'w') as out:
                json.dump(self._trigram_vocabulary, out, indent=2)
        self._p['vocab.size'] = len(self._trigram_vocabulary)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        word_input = keras.layers.Input(shape=(self._p['max.sent.len'], self._p['vocab.size'],), dtype='float32', name='sentence_input')
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(word_input)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[word_input], output=[semantic_vector], name=self._sibling_model_name)
        self.logger.debug("Sibling model is finished.")
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],  self._p['vocab.size'],), dtype='float32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],  self._p['vocab.size'],), dtype='float32',
                                        name='edge_input')
        # Twins model
        sentence_vector = sibiling_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(sibiling_model)(edge_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, edge_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def encode_data_instance(self, instance):
        sentence_encoded, edges_encoded = self.encode_by_trigram(instance)
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        edges_ids = sequence.pad_sequences(edges_encoded, maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        return sentence_ids, edges_ids

    def encode_by_trigram(self, graph_set):
        sentence_tokens = graph_set[0].get("tokens", [])
        sentence_trigrams = [set(string_to_trigrams(token)) for token in sentence_tokens]
        sentence_encoded = [[int(t in trigrams) for t in self._trigram_vocabulary]
                            for trigrams in sentence_trigrams]
        edges_encoded = []
        for g in graph_set:
            first_edge = graph.get_graph_first_edge(g)
            property_label = first_edge.get('label', '')
            edge_trigrams = [set(string_to_trigrams(token)) for token in property_label.split()]
            edge_encoded = [[int(t in trigrams) for t in self._trigram_vocabulary]
                            for trigrams in edge_trigrams]
            edges_encoded.append(edge_encoded)
        return sentence_encoded, edges_encoded

    def encode_batch_by_trigrams(self, graphs, verbose=False):
        graphs = [el for el in graphs if el]
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        edges_matrix = np.zeros((len(graphs), len(graphs[0]),  self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")

        for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, edges_encoded = self.encode_by_trigram(graph_set)
            assert len(edges_encoded) == edges_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, edge_encoded in enumerate(edges_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                edges_matrix[index, i, :len(edge_encoded)] = edge_encoded

        return sentences_matrix, edges_matrix

    def load_from_file(self, path_to_model):
        super(YihModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: trigram_vocabulary_{}.json".format(self._model_number))
        with open(self._save_model_to + "trigram_vocabulary_{}.json".format(self._model_number)) as f:
            self._trigram_vocabulary = json.load(f)
        self._trigram_vocabulary = [tuple(t) for t in self._trigram_vocabulary]
        self._p['vocab.size'] = len(self._trigram_vocabulary)
        self.logger.debug("Vocabulary size: {}.".format(len(self._trigram_vocabulary)))


class TrigramCNNEdgeSumModel(BrothersModel, YihModel):

    def prepare_model(self, train_tokens, properties_set):
        YihModel.extract_vocabulary(self, train_tokens)
        BrothersModel.prepare_model(self, train_tokens, properties_set)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")

        #Graph Model
        edge_input = keras.layers.Input(shape=(self._p['max.graph.size'], self._p['max.sent.len'],
                                               self._p['vocab.size'],), dtype='float32', name='edge_input')
        edge_vectors = keras.layers.TimeDistributed(self._get_sibling_model())(edge_input)
        graph_vector = keras.layers.GlobalMaxPooling1D()(edge_vectors)
        graph_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                          activation=self._p.get("sibling.activation", 'tanh'),
                                          init=self._p.get("sibling.weight.init", 'glorot_uniform'))(graph_vector)
        graph_model = keras.models.Model(input=[edge_input], output=[graph_vector])
        self.logger.debug("Graph model is finished: {}".format(graph_model))

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='sentence_input')
        graph_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.graph.size'],
                                               self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='graph_input')
        sentence_vector = self._get_sibling_model()(sentence_input)
        graph_vectors = keras.layers.TimeDistributed(graph_model, name=self._younger_model_name)(graph_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, graph_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, graph_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def _get_sibling_model(self):
        # Sibling model
        if self._sibling_model and self._p.get('sibling.singleton', False):
            return self._sibling_model
        word_input = keras.layers.Input(shape=(self._p['max.sent.len'], self._p['vocab.size'],), dtype='float32',
                                        name='sentence_input')

        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(word_input)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)
        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        if self._p.get("relu.on.top", False):
            semantic_vector = keras.layers.Activation('relu')(semantic_vector)
        sibiling_model = keras.models.Model(input=[word_input], output=[semantic_vector], name=self._older_model_name)
        self.logger.debug("Sibling model is finished.")
        self._sibling_model = sibiling_model
        return sibiling_model

    def encode_data_instance(self, instance):
        sentence_encoded, graphs_encoded = self.encode_by_trigram(instance)
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graph_matrix = np.zeros((len(graphs_encoded), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_by_trigram(self, graph_set):
        sentence_tokens = graph_set[0].get("tokens", [])
        sentence_trigrams = [set(string_to_trigrams(token)) for token in sentence_tokens]
        sentence_encoded = [[int(t in trigrams) for t in self._trigram_vocabulary]
                            for trigrams in sentence_trigrams]
        graphs_encoded = []
        for g in graph_set:
            edges_encoded = []
            for edge in g.get('edgeSet', []):
                property_label = edge.get('label', '')
                edge_trigrams = [set(string_to_trigrams(token)) for token in property_label.split()]
                edge_encoded = [[int(t in trigrams) for t in self._trigram_vocabulary]
                                for trigrams in edge_trigrams]
                edges_encoded.append(edge_encoded)
            graphs_encoded.append(edges_encoded)
        return sentence_encoded, graphs_encoded

    def encode_batch_by_trigrams(self, graphs, verbose=False):
        graphs = [el for el in graphs if el]
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")
        graph_matrix = np.zeros((len(graphs), len(graphs[0]), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int8")

        for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, graphs_encoded = self.encode_by_trigram(graph_set)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_encoded in enumerate(graph_encoded):
                    edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                    graph_matrix[index, i, j, :len(edge_encoded)] = edge_encoded
        return sentences_matrix, graph_matrix


class TrigramCNNGraphModel(TrigramCNNEdgeSumModel):

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='sentence_input')
        graph_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.graph.size'],
                                                self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='graph_input')
        sentence_vector = self._get_sibling_model()(sentence_input)
        graph_vectors = keras.layers.TimeDistributed(self._get_graph_model(), name=self._younger_model_name)(graph_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, graph_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, graph_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def _get_graph_model(self):
        #Graph Model
        # Encode edges with property emebddings
        # Encode filter,
        edge_input = keras.layers.Input(shape=(self._p['max.graph.size'], self._p['max.sent.len'],
                                               self._p['vocab.size'],), dtype='float32', name='edge_input')
        edge_vectors = keras.layers.TimeDistributed(self._get_sibling_model())(edge_input)
        if self._p.get("graph.sum", 'sum') == 'sum':
            graph_vector = keras.layers.Lambda(lambda x: K.sum(x, axis=1),
                                              output_shape=(self._p['sem.layer.size'],))(edge_vectors)
        else:
            graph_vector = keras.layers.GlobalMaxPooling1D()(edge_vectors)
        graph_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                          activation=self._p.get("sibling.activation", 'tanh'),
                                          init=self._p.get("sibling.weight.init", 'glorot_uniform'))(graph_vector)
        graph_model = keras.models.Model(input=[edge_input], output=[graph_vector])
        self.logger.debug("Graph model is finished: {}".format(graph_model))
        return graph_model


class TrigramCNNGraphSymbolicModel(TrigramCNNEdgeSumModel):

    def __init__(self, **kwargs):
        self._property2idx = {utils.all_zeroes: 0, utils.unknown_el: 1}
        self._propertytype2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "v": 2, "q": 3}
        self._type2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "direct": 2, "reverse": 3, "v-structure": 4, "time": 5}
        self._modifier2idx = {utils.all_zeroes: 0, utils.unknown_el: 1, "argmax": 2, "argmin": 3, "num": 4, "filter": 5}
        super(TrigramCNNGraphSymbolicModel, self).__init__(**kwargs)

    def prepare_model(self, train_tokens, properties_set):
        YihModel.extract_vocabulary(self, train_tokens)
        self.init_property_index(properties_set)
        WordCNNModel.extract_vocabualry(self, train_tokens)
        BrothersModel.prepare_model(self, train_tokens, properties_set)

    def init_property_index(self, properties_set):
        properties_set = properties_set | wdaccess.HOP_UP_RELATIONS | wdaccess.HOP_DOWN_RELATIONS
        self._property2idx.update({p:i for i, p in enumerate(properties_set, start=len(self._property2idx))})
        self.logger.debug("Property index is finished: {}".format(len(self._property2idx)))
        with open(self._save_model_to + "property2idx_{}.json".format(self._model_number), 'w') as out:
            json.dump(self._property2idx, out, indent=2)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='sentence_input')

        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p.get('max.graph.size', 3), 6), dtype='int32', name='edge_input')

        sentence_vector = self._get_sibling_model()(sentence_input)
        graph_vectors = keras.layers.TimeDistributed(self._get_graph_model(), name=self._younger_model_name)(edge_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, graph_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def _get_embedding_model(self, input_shape, emb_dim, vocab_size):
        e_input = keras.layers.Input(shape=input_shape, dtype='float32', name='e_input')
        embeddings_layer = keras.layers.Embedding(output_dim=emb_dim, input_dim=vocab_size,
                                                 input_length=input_shape[-1], init=self._p.get("emb.weight.init", 'uniform'),
                                                       trainable=True)
        if len(input_shape) > 1:
            embeddings = keras.layers.TimeDistributed(embeddings_layer)(e_input)
            embeddings = keras.layers.TimeDistributed(keras.layers.Flatten())(embeddings)
        else:
            embeddings = embeddings_layer(e_input)
        return keras.models.Model(input=[e_input], output=[embeddings])

    def _get_graph_model(self):
        edge_input = keras.layers.Input(shape=(self._p.get('max.graph.size', 3), 6 + self._p.get('max.label.len', 5) * self._p['vocab.size']), dtype='float32', name='edge_input')

        kbid_input = keras.layers.Lambda(lambda i: i[:, :, :4], output_shape=(self._p.get('max.graph.size', 3), 4))(edge_input)
        type_input = keras.layers.Lambda(lambda i: i[:, :, 4], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
        rel_type_input = keras.layers.Lambda(lambda i: i[:, :, 5], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)

        kbid_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3), 4), emb_dim=self._p['property.emb.dim'], vocab_size=len(self._property2idx))
        
        type_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['type.emb.dim'], vocab_size=len(self._type2idx))
        rel_type_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['ptype.emb.dim'], vocab_size=len(self._propertytype2idx))
        
        kbid_embeddings = kbid_embeddings_layer(kbid_input)
        type_embeddings = type_embeddings_layer(type_input)
        rel_type_embeddings = rel_type_embeddings_layer(rel_type_input)

        edge_vectors = keras.layers.Merge(mode='concat')([kbid_embeddings, type_embeddings, rel_type_embeddings])
        edge_vectors = keras.layers.TimeDistributed(
            keras.layers.Dense(self._p['sem.layer.size'],
                               activation=self._p.get("sibling.activation", 'tanh'),
                               init=self._p.get("sibling.weight.init", 'glorot_uniform')))(edge_vectors)
        edge_vectors = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(edge_vectors)
        if self._p.get("graph.sum", 'sum') == 'sum':
            graph_vector = keras.layers.Lambda(lambda x: K.sum(x, axis=1),
                                               output_shape=(self._p['sem.layer.size'],))(edge_vectors)
        else:
            graph_vector = keras.layers.GlobalMaxPooling1D()(edge_vectors)  

        graph_vector = keras.layers.Dense(self._p['sem.layer.size'],
                            activation=self._p.get("sibling.activation", 'tanh'),
                            init=self._p.get("sibling.weight.init", 'glorot_uniform'))(graph_vector)
        graph_vector = keras.layers.Dropout(self._p['dropout.sibling'])(graph_vector)
        graph_model = keras.models.Model(input=[edge_input], output=[graph_vector])
        self.logger.debug("Graph model is finished: {}".format(graph_model))
        return graph_model

    def encode_data_instance(self, instance):
        sentence_encoded, _ = TrigramCNNEdgeSumModel.encode_data_instance(self, instance[:1])
        graph_matrix = np.zeros((len(instance), self._p.get('max.graph.size', 3), 6), dtype="int32")
        for i, g in enumerate(instance):
            for j, edge in enumerate(g.get("edgeSet", [])[:self._p.get('max.graph.size', 3)]):
                if edge.get('type') != 'time':
                    edge_kbid = edge.get('kbID')[:-1] if 'kbID' in edge else utils.unknown_el
                else:
                    edge_kbid = "argmax" if "argmax" in edge else "argmin"
                graph_matrix[i, j] = [
                    self._property2idx.get(edge_kbid, 0),
                    self._property2idx.get(edge['hopUp'][:-1] if 'hopUp' in edge else utils.all_zeroes, 0),
                    self._property2idx.get(edge['hopDown'][:-1] if 'hopDown' in edge else utils.all_zeroes, 0),
                    self._property2idx.get("argmax" if "argmax" in edge
                                           else "argmin" if "argmin" in edge
                                            else "num" if "num" in edge
                                            else "filter" if "filter" in edge
                                            else utils.all_zeroes, 0),
                    self._type2idx.get(edge.get('type', utils.unknown_el), 0),
                    self._propertytype2idx.get(edge['kbID'][-1] if 'kbID' in edge else utils.unknown_el, 0),
                ]
        return sentence_encoded, graph_matrix

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))

        sentences_matrix = np.zeros((len(input_set), self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int32")
        graph_matrix = np.zeros((len(input_set), len(input_set[0]), self._p.get('max.graph.size', 3), 6), dtype="int32")
        for s in range(len(input_set)):
            sentence_encoded, _ = self.encode_by_trigram(input_set[s][:1])
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[s, :len(sentence_encoded)] = sentence_encoded

            for i, g in enumerate(input_set[s]):
                for j, edge in enumerate(g.get("edgeSet", [])[:self._p.get('max.graph.size', 3)]):
                    if edge.get('type') != 'time':
                        edge_kbid = edge.get('kbID')[:-1] if 'kbID' in edge else utils.unknown_el
                    else:
                        edge_kbid = "argmax" if "argmax" in edge else "argmin"
                    graph_matrix[s, i, j] = [
                        self._property2idx.get(edge_kbid, 0),
                        self._property2idx.get(edge['hopUp'][:-1] if 'hopUp' in edge else utils.all_zeroes, 0),
                        self._property2idx.get(edge['hopDown'][:-1] if 'hopDown' in edge else utils.all_zeroes, 0),
                        self._property2idx.get("argmax" if "argmax" in edge
                                               else "argmin" if "argmin" in edge
                                                else "num" if "num" in edge
                                                else "filter" if "filter" in edge
                                                else utils.all_zeroes, 0),
                        self._type2idx.get(edge.get('type', utils.unknown_el), 0),
                        self._propertytype2idx.get(edge['kbID'][-1] if 'kbID' in edge else utils.unknown_el, 0),
                    ]
        assert len(sentences_matrix) == len(graph_matrix) == len(targets)
        return sentences_matrix, graph_matrix, targets

    def load_from_file(self, path_to_model):
        super(TrigramCNNGraphSymbolicModel, self).load_from_file(path_to_model=path_to_model)

        self.logger.debug("Loading property index from: property2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "property2idx_{}.json".format(self._model_number)) as f:
            self._property2idx = json.load(f)
        self.logger.debug("Vocabulary size: {}.".format(len(self._property2idx)))


class TrigramCNNGraphSymbolicWithEmbModel(TrigramCNNGraphSymbolicModel, WordCNNModel):

    def __init__(self, **kwargs):
        super(TrigramCNNGraphSymbolicWithEmbModel, self).__init__(**kwargs)
        self._feature_vector_size = sum(int(v) if v else 1 for f, v in self._p.get('symbolic.features', {}).items())
        self.logger.debug("Feature vector size: {}".format(self._feature_vector_size))

    def prepare_model(self, train_tokens, properties_set):
        YihModel.extract_vocabulary(self, train_tokens)
        TrigramCNNGraphSymbolicModel.init_property_index(self, properties_set)
        WordCNNModel.extract_vocabualry(self, train_tokens)
        BrothersModel.prepare_model(self, train_tokens, properties_set)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],  self._p['vocab.size']), dtype='float32', name='sentence_input')

        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p.get('max.graph.size', 3), self._feature_vector_size), dtype='int32', name='edge_input')

        sentence_vector = self._get_sibling_model()(sentence_input)
        graph_vectors = keras.layers.TimeDistributed(self._get_graph_model(), name=self._younger_model_name)(edge_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, graph_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer=keras.optimizers.Adam(clipnorm=1.), loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def _get_graph_model(self):
        edge_input = keras.layers.Input(shape=(self._p.get('max.graph.size', 3), self._feature_vector_size), dtype='float32', name='edge_input')

        kbid_input = keras.layers.Lambda(lambda i: i[:, :, 0], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
        kbid_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['property.emb.dim'], vocab_size=len(self._property2idx))
        layers_to_concat = []
        kbid_embeddings = kbid_embeddings_layer(kbid_input)
        layers_to_concat.append(kbid_embeddings)

        if self._p.get('symbolic.features', {}).get("hopUp", False):
            hopUp_input = keras.layers.Lambda(lambda i: i[:, :, 1], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
            hopUp_embeddings = kbid_embeddings_layer(hopUp_input)
            layers_to_concat.append(hopUp_embeddings)

        if self._p.get('symbolic.features', {}).get("hopDown", False):
            hopDown_input = keras.layers.Lambda(lambda i: i[:, :, 2], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
            hopDown_embeddings = kbid_embeddings_layer(hopDown_input)
            layers_to_concat.append(hopDown_embeddings)

        if self._p.get('symbolic.features', {}).get("modifier", False):
            modifier_input = keras.layers.Lambda(lambda i: i[:, :, 3], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
            modifier_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['type.emb.dim'], vocab_size=len(self._modifier2idx))
            modifier_embeddings = modifier_embeddings_layer(modifier_input)
            layers_to_concat.append(modifier_embeddings)

        if self._p.get('symbolic.features', {}).get("rel.type", False):
            type_input = keras.layers.Lambda(lambda i: i[:, :, 4], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
            type_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['type.emb.dim'], vocab_size=len(self._type2idx))
            type_embeddings = type_embeddings_layer(type_input)
            layers_to_concat.append(type_embeddings)

        if self._p.get('symbolic.features', {}).get("property.type", False):
            property_type_input = keras.layers.Lambda(lambda i: i[:, :, 5], output_shape=(self._p.get('max.graph.size', 3),))(edge_input)
            property_type_embeddings_layer = self._get_embedding_model(input_shape=(self._p.get('max.graph.size', 3),), emb_dim=self._p['type.emb.dim'], vocab_size=len(self._propertytype2idx))
            property_type_embeddings = property_type_embeddings_layer(property_type_input)
            layers_to_concat.append(property_type_embeddings)

        if self._p.get('symbolic.features', {}).get("right.label", 0) > 0:
            right_label_input = keras.layers.Lambda(lambda i: i[:, :, 6:],
                                                    output_shape=(self._p.get('max.graph.size', 3), self._p.get('symbolic.features', {}).get("right.label", 5)))(edge_input)
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings_layer = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                           input_dim=self._embedding_matrix.shape[0],
                                                           input_length=self._p.get('symbolic.features', {}).get("right.label", 5),
                                                           weights=[self._embedding_matrix],
                                                           trainable=False,
                                                           mask_zero=False)
            word_embeddings = keras.layers.TimeDistributed(word_embeddings_layer)(right_label_input)
            word_embeddings = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling1D())(word_embeddings)
            layers_to_concat.append(word_embeddings)

        edge_vectors = keras.layers.Merge(mode='concat')(layers_to_concat)
        edge_vectors = keras.layers.TimeDistributed(
            keras.layers.Dense(self._p['sem.layer.size'],
                               activation=self._p.get("sibling.activation", 'tanh'),
                               init=self._p.get("sibling.weight.init", 'glorot_uniform')))(edge_vectors)
        edge_vectors = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(edge_vectors)
        if self._p.get("graph.sum", 'sum') == 'sum':
            graph_vector = keras.layers.Lambda(lambda x: K.sum(x, axis=1),
                                               output_shape=(self._p['sem.layer.size'],))(edge_vectors)
        else:
            graph_vector = keras.layers.GlobalMaxPooling1D()(edge_vectors)

        graph_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                          activation=self._p.get("sibling.activation", 'tanh'),
                                          init=self._p.get("sibling.weight.init", 'glorot_uniform'))(graph_vector)
        graph_vector = keras.layers.Dropout(self._p['dropout.sibling'])(graph_vector)
        if self._p.get("relu.on.top", False):
            graph_vector = keras.layers.Activation('relu')(graph_vector)
        graph_model = keras.models.Model(input=[edge_input], output=[graph_vector])
        self.logger.debug("Graph model is finished: {}".format(graph_model))
        return graph_model

    def encode_data_instance(self, instance):
        sentence_encoded, _ = self.encode_by_trigram(instance[:1])
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graph_matrix = np.zeros((len(instance), self._p.get('max.graph.size', 3), self._feature_vector_size), dtype="int32")
        for i, g in enumerate(instance):
            for j, edge in enumerate(g.get("edgeSet", [])[:self._p.get('max.graph.size', 3)]):
                edge_feature_vector = self.get_edge_feature_vector(edge)
                graph_matrix[i, j, :len(edge_feature_vector)] = edge_feature_vector
        return sentence_ids, graph_matrix

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
                          right_label_ids]
        assert len(feature_vector) <= self._feature_vector_size
        return feature_vector

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))

        sentences_matrix = np.zeros((len(input_set), self._p.get('max.sent.len', 10), len(self._trigram_vocabulary)), dtype="int32")
        graph_matrix = np.zeros((len(input_set), len(input_set[0]), self._p.get('max.graph.size', 3), self._feature_vector_size), dtype="int32")
        for s in range(len(input_set)):
            sentence_encoded, _ = self.encode_by_trigram(input_set[s][:1])
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[s, :len(sentence_encoded)] = sentence_encoded

            for i, g in enumerate(input_set[s]):
                for j, edge in enumerate(g.get("edgeSet", [])[:self._p.get('max.graph.size', 3)]):
                    edge_feature_vector = self.get_edge_feature_vector(edge)
                    graph_matrix[s, i, j, :len(edge_feature_vector)] = edge_feature_vector
        return sentences_matrix, graph_matrix, targets


def string_to_unigrams(input_string, character2idx):
    return [character2idx.get(c, character2idx[utils.unknown_el]) for c in input_string]


def string_to_trigrams(t):
    """
    Convert a token to a list of trigrams following the hashing technique.

    :param t: a single token as a string
    :return: list of triples of characters
    >>> list(string_to_trigrams('who'))
    [('#', 'w', 'h'), ('w', 'h', 'o'), ('h', 'o', '#')]
    """
    return nltk.ngrams("#{}#".format(t), 3)


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
