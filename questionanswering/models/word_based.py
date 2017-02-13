from collections import defaultdict
import keras
import tqdm
from keras import backend as K
from keras.preprocessing import sequence
import numpy as np
import json
import utils
from construction import graph

from models.kerasmodel import TwinsModel, BrothersModel
from models import keras_extensions


class WordCNNModel(TwinsModel):

    def __init__(self, **kwargs):
        self._word2idx = defaultdict(int)
        self._embedding_matrix = None
        super(WordCNNModel, self).__init__(**kwargs)

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = self.encode_batch_by_tokens(input_set, verbose=False)
        if self._p.get("loss", 'categorical_crossentropy') == 'categorical_crossentropy':
            targets = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets

    def prepare_model(self, train_tokens, properties_set):
        self.extract_vocabualry(train_tokens)
        super(WordCNNModel, self).prepare_model(train_tokens)

    def extract_vocabualry(self, train_tokens):
        if not self._word2idx:
            if "word.embeddings" in self._p:
                self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
                self.logger.debug('Word index loaded, size: {}'.format(len(self._word2idx)))
            else:
                self._word2idx = utils.get_word_index([t for tokens in train_tokens for t in tokens])
                self.logger.debug('Word index created, size: {}'.format(len(self._word2idx)))
                with open(self._save_model_to + "word2idx_{}.json".format(self._model_number), 'w') as out:
                    json.dump(self._word2idx, out, indent=2)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        if "word.embeddings" in self._p:
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                     input_dim=self._embedding_matrix.shape[0],
                                                     input_length=self._p['max.sent.len'],
                                                     weights=[self._embedding_matrix],
                                                     mask_zero=False, trainable=self._p.get("emb.train", False))(tokens_input)
        else:
            word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=len(self._word2idx),
                                                     input_length=self._p['max.sent.len'],
                                                     init=self._p.get("emb.weight.init", 'uniform'),
                                                     mask_zero=False)(tokens_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(word_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[tokens_input], output=[semantic_vector], name=self._sibling_model_name)
        self.logger.debug("Sibling model is finished.")

        # Twins model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int32',
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

    def encode_data_instance(self, instance):
        sentence_encoded, edges_encoded = self.encode_by_tokens(instance)
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        edges_ids = sequence.pad_sequences(edges_encoded, maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        return sentence_ids, edges_ids

    def encode_by_tokens(self, graph_set):
        sentence_tokens = graph_set[0].get("tokens", [])
        sentence_encoded = [utils.get_idx(t, self._word2idx) for t in sentence_tokens]
        edges_encoded = []
        for g in graph_set:
            first_edge = graph.get_graph_first_edge(g)
            property_label = first_edge.get('label', '')
            edge_ids = [utils.get_idx(t, self._word2idx) for t in property_label.split()]
            edges_encoded.append(edge_ids)

        return sentence_encoded, edges_encoded

    def encode_batch_by_tokens(self, graphs, verbose=False):
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10)), dtype="int32")
        edges_matrix = np.zeros((len(graphs), len(graphs[0]), self._p.get('max.sent.len', 10)), dtype="int32")

        for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, edges_encoded = self.encode_by_tokens(graph_set)
            assert len(edges_encoded) == edges_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, edge_encoded in enumerate(edges_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                edges_matrix[index, i, :len(edge_encoded)] = edge_encoded

        return sentences_matrix, edges_matrix

    def load_from_file(self, path_to_model):
        super(WordCNNModel, self).load_from_file(path_to_model=path_to_model)

        if "word.embeddings" in self._p:
            self.logger.debug("Loading pre-trained word embeddings.")
            self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
        else:
            self.logger.debug("Loading vocabulary from: word2idx_{}.json".format(self._model_number))
            with open(self._save_model_to + "word2idx_{}.json".format(self._model_number)) as f:
                self._word2idx = json.load(f)
        self.logger.debug("Vocabulary size: {}.".format(len(self._word2idx)))


class WordGraphModel(BrothersModel, WordCNNModel):

    def prepare_model(self, train_tokens, properties_set):
        WordCNNModel.extract_vocabualry(self, train_tokens)
        BrothersModel.prepare_model(self, train_tokens, properties_set)

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='float32', name='sentence_input')
        graph_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.graph.size'],
                                                self._p['max.sent.len']), dtype='float32', name='graph_input')
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

    def _get_sibling_model(self):
        # Sibling model
        if self._sibling_model and self._p.get('sibling.singleton', False):
            return self._sibling_model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        if "word.embeddings" in self._p:
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                     input_dim=self._embedding_matrix.shape[0],
                                                     input_length=self._p['max.sent.len'],
                                                     weights=[self._embedding_matrix],
                                                     mask_zero=False, trainable=self._p.get("emb.train", False))(tokens_input)
        else:
            word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=len(self._word2idx),
                                                     input_length=self._p['max.sent.len'],
                                                     init=self._p.get("emb.weight.init", 'uniform'),
                                                     mask_zero=False)(tokens_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(word_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)
        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(
                semantic_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[tokens_input], output=[semantic_vector], name=self._older_model_name)
        self.logger.debug("Sibling model is finished.")
        return sibiling_model

    def _get_graph_model(self):
        edge_input = keras.layers.Input(shape=(self._p['max.graph.size'], self._p['max.sent.len'],), dtype='float32', name='edge_input')
        edge_vectors = keras.layers.TimeDistributed(self._get_sibling_model())(edge_input)
        if self._p.get("graph.sum", 'sum') == 'sum':
            graph_vector = keras.layers.Lambda(lambda x: K.sum(x, axis=1),
                                               output_shape=(self._p['sem.layer.size'],))(edge_vectors)
        else:
            graph_vector = keras.layers.GlobalMaxPooling1D()(edge_vectors)

        if self._p.get('graph.dense.layer', False):
            graph_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                              activation=self._p.get("sibling.activation", 'tanh'),
                                              init=self._p.get("sibling.weight.init", 'glorot_uniform'))(graph_vector)

        graph_model = keras.models.Model(input=[edge_input], output=[graph_vector])
        self.logger.debug("Graph model is finished: {}".format(graph_model))
        return graph_model

    def encode_data_instance(self, instance):
        sentence_encoded, graphs_encoded = self.encode_by_tokens(instance)
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        graph_matrix = np.zeros((len(graphs_encoded), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10)), dtype="int8")
        for i, graph_encoded in enumerate(graphs_encoded):
            graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
            for j, edge_encoded in enumerate(graph_encoded):
                edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                graph_matrix[i, j, :len(edge_encoded)] = edge_encoded
        return sentence_ids, graph_matrix

    def encode_by_tokens(self, graph_set):
        sentence_tokens = graph_set[0].get("tokens", [])
        sentence_encoded = [utils.get_idx(t, self._word2idx) for t in sentence_tokens]
        graphs_encoded = []
        for g in graph_set:
            edges_encoded = []
            for edge in g.get('edgeSet', []):
                property_label = edge.get('label', '')
                edge_ids = [utils.get_idx(t, self._word2idx) for t in property_label.split()]
                edges_encoded.append(edge_ids)
            graphs_encoded.append(edges_encoded)
        return sentence_encoded, graphs_encoded

    def encode_batch_by_tokens(self, graphs, verbose=False):
        sentences_matrix = np.zeros((len(graphs), self._p.get('max.sent.len', 10)), dtype="int32")
        graph_matrix = np.zeros((len(graphs), len(graphs[0]), self._p.get('max.graph.size', 3),
                                 self._p.get('max.sent.len', 10)), dtype="int8")

        for index, graph_set in enumerate(tqdm.tqdm(graphs, ascii=True, disable=(not verbose))):
            sentence_encoded, graphs_encoded = self.encode_by_tokens(graph_set)
            assert len(graphs_encoded) == graph_matrix.shape[1]
            sentence_encoded = sentence_encoded[:self._p.get('max.sent.len', 10)]
            sentences_matrix[index, :len(sentence_encoded)] = sentence_encoded
            for i, graph_encoded in enumerate(graphs_encoded):
                graph_encoded = graph_encoded[:self._p.get('max.graph.size', 3)]
                for j, edge_encoded in enumerate(graph_encoded):
                    edge_encoded = edge_encoded[:self._p.get('max.sent.len', 10)]
                    graph_matrix[index, i, j, :len(edge_encoded)] = edge_encoded
        return sentences_matrix, graph_matrix


class WordSumModel(WordCNNModel):

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        if "word.embeddings" in self._p:
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                     input_dim=self._embedding_matrix.shape[0],
                                                     input_length=self._p['max.sent.len'],
                                                     weights=self._embedding_matrix,
                                                     mask_zero=False, trainable=self._p.get("emb.train", False))(tokens_input)
        else:
            word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=len(self._word2idx),
                                                     input_length=self._p['max.sent.len'],
                                                     init=self._p.get("emb.weight.init", 'uniform'),
                                                     mask_zero=False)(tokens_input)
        if self._p.get("emb.sum", False):
            semantic_vector = keras.layers.Lambda(lambda x: K.sum(x, axis=1),
                                          output_shape=(self._embedding_matrix.shape[1] if "word.embeddings" in self._p else
                                                        self._p['emb.dim'],))(word_embeddings)
        else:
            semantic_vector = keras.layers.GlobalMaxPooling1D()(word_embeddings)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[tokens_input], output=[semantic_vector], name=self._sibling_model_name)
        self.logger.debug("Sibling model is finished.")

        # Twins model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int32',
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


class WordCNNBrotherModel(BrothersModel, WordCNNModel):

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Older model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32')
        word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=len(self._word2idx),
                                                 input_length=self._p['max.sent.len'],
                                                 init=self._p.get("emb.weight.init", 'uniform'),
                                                 mask_zero=False)(tokens_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same',
                                                     init=self._p.get("sibling.weight.init", 'glorot_uniform'))(word_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'),
                                                 init=self._p.get("sibling.weight.init", 'glorot_uniform'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        older_model = keras.models.Model(input=[tokens_input], output=[semantic_vector], name=self._older_model_name)
        self.logger.debug("Older model is finished: {}.".format(older_model))

        # Younger model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32')
        word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=len(self._word2idx),
                                                 input_length=self._p['max.sent.len'],
                                                 mask_zero=False)(tokens_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same')(word_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'))(semantic_vector)

        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        younger_model = keras.models.Model(input=[tokens_input], output=[semantic_vector], name=self._younger_model_name)
        self.logger.debug("Younger model is finished: {}.".format(younger_model))

        # Brothers model
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int32',
                                        name='edge_input')

        sentence_vector = older_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(younger_model, name=self._younger_model_name)(edge_input)

        main_output = keras.layers.Merge(mode=keras_extensions.keras_cosine if self._p.get("twin.similarity") == 'cos' else self._p.get("twin.similarity", 'dot'),
                                         dot_axes=(1, 2), name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, edge_vectors])

        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss=self._p.get("loss", 'categorical_crossentropy'), metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def prepare_model(self, train_tokens):
        WordCNNModel.extract_vocabualry(self, train_tokens)
        BrothersModel.prepare_model(self, train_tokens)
