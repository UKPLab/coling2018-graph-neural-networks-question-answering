from collections import defaultdict
import keras
from keras import backend as K
from keras.preprocessing import sequence
import numpy as np
import json
import re
import utils

from .qamodel import TwinsModel
from . import input_to_indices
from wikidata import wdaccess
from . import keras_extensions


class WordCNNModel(TwinsModel):

    def __init__(self, **kwargs):
        self._word2idx = defaultdict(int)
        self._embedding_matrix = None
        super(WordCNNModel, self).__init__(**kwargs)

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = input_to_indices.encode_batch_by_tokens(input_set, self._word2idx,
                                                                                 wdaccess.property2label,
                                                                                 max_input_len=self._p.get('max.sent.len', 10), verbose=True)
        targets_as_one_hot = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets_as_one_hot

    def train(self, data_with_targets, validation_with_targets=None):
        if not self._word2idx:
            if "word.embeddings" in self._p:
                self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
            else:
                self._word2idx = input_to_indices.get_word_index([t for graphs in data_with_targets[0]
                                                                  for t in graphs[0].get('tokens', []) if graphs])
                self.logger.debug('Word index created, size: {}'.format(len(self._word2idx)))
                with open(self._save_model_to + "word2idx_{}.json".format(self._model_number), 'w') as out:
                    json.dump(self._word2idx, out, indent=2)
        self._p['vocab.size'] = len(self._word2idx)

        super(WordCNNModel, self).train(data_with_targets, validation_with_targets)
        self._sibling_model = self._model.get_layer(name="sibiling_model")

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        if "word.embeddings" in self._p:
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                     input_dim=self._p['vocab.size'],
                                                     input_length=self._p['max.sent.len'],
                                                     mask_zero=False, trainable=False)(tokens_input)
        else:
            word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=self._p['vocab.size'],
                                                     input_length=self._p['max.sent.len'],
                                                     mask_zero=False)(tokens_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same')(word_embeddings)
        semantic_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling.pooling'])(semantic_vector)

        for i in range(self._p.get("sem.layer.depth", 1)):
            semantic_vector = keras.layers.Dense(self._p['sem.layer.size'],
                                                 activation=self._p.get("sibling.activation", 'tanh'))(semantic_vector)

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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def encode_data_instance(self, instance):
        sentence_encoded, edges_encoded = input_to_indices.encode_by_tokens(instance, self._word2idx, wdaccess.property2label)
        sentence_ids = sequence.pad_sequences([sentence_encoded], maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        edges_ids = sequence.pad_sequences(edges_encoded, maxlen=self._p.get('max.sent.len', 10), padding='post', truncating='post', dtype="int32")
        return sentence_ids, edges_ids

    def load_from_file(self, path_to_model):
        super(WordCNNModel, self).load_from_file(path_to_model=path_to_model)

        if "word.embeddings" in self._p:
            self.logger.debug("Loading pre-trained word embeddings.")
            self._embedding_matrix, self._word2idx = utils.load(self._p['word.embeddings'])
        else:
            self.logger.debug("Loading vocabulary from: word2idx_{}.json".format(self._model_number))
            with open(self._save_model_to + "word2idx_{}.json".format(self._model_number)) as f:
                self._word2idx = json.load(f)
        self._p['vocab.size'] = len(self._word2idx)
        self.logger.debug("Vocabulary size: {}.".format(len(self._word2idx)))


class WordSumModel(WordCNNModel):

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        # Sibling model
        tokens_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        if "word.embeddings" in self._p:
            self.logger.debug("Using a pre-trained embedding matrix.")
            word_embeddings = keras.layers.Embedding(output_dim=self._embedding_matrix.shape[1],
                                                     input_dim=self._p['vocab.size'],
                                                     input_length=self._p['max.sent.len'],
                                                     mask_zero=False, trainable=False)(tokens_input)
        else:
            word_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=self._p['vocab.size'],
                                                     input_length=self._p['max.sent.len'],
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
                                                 activation=self._p.get("sibling.activation", 'tanh'))(semantic_vector)

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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

