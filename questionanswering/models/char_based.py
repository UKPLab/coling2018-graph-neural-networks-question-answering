from collections import defaultdict
import keras
from keras import backend as K
from keras.preprocessing import sequence
import numpy as np
import json
import re

from .qamodel import TwinsModel
from . import input_to_indices
from wikidata import wdaccess


class CharCNNModel(TwinsModel):
    def __init__(self, **kwargs):
        self._sibling_model = None
        self._character2idx = defaultdict(int)
        super(CharCNNModel, self).__init__(**kwargs)

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sibling_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._sibling_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))

        return np.argsort(predictions)[::-1]

    def encode_data_instance(self, instance, **kwargs):
        sentence_ids, edges_ids = input_to_indices.encode_by_character(instance, self._character2idx,
                                                                       wdaccess.property2label,
                                                                       edge_with_entity=self._p.get('edge.with.entity', False))
        sentence_ids = sequence.pad_sequences([sentence_ids], maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int32")
        edges_ids = sequence.pad_sequences(edges_ids, maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int32")
        return sentence_ids, edges_ids

    def train(self, data_with_targets, validation_with_targets=None):
        if not self._character2idx:
            self._character2idx = input_to_indices.get_character_index(
                [" ".join(graphs[0]['tokens']) for graphs in data_with_targets[0] if graphs])
            self.logger.debug('Character index created, size: {}'.format(len(self._character2idx)))
            with open(self._save_model_to + "character2idx_{}.json".format(self._model_number), 'w') as out:
                json.dump(self._character2idx, out, indent=2)
        self._p['vocab.size'] = len(self._character2idx)

        super(CharCNNModel, self).train(self, data_with_targets, validation_with_targets)
        self._sibling_model = self._model.get_layer(name="sibiling_model")

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        characters_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        character_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=self._p['vocab.size'],
                                                      input_length=self._p['max.sent.len'],
                                                      mask_zero=False)(characters_input)
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same')(character_embeddings)
        sentence_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)

        # semantic_vector = keras.layers.Dense(self._p['sem.layer.size'] * 3, activation='tanh')(sentence_vector)
        semantic_vector = keras.layers.Dense(self._p['sem.layer.size'], activation='tanh', name='semantic_vector')(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        sibiling_model = keras.models.Model(input=[characters_input], output=[semantic_vector], name=self._sibling_model_name)
        self.logger.debug("Sibling model is finished.")
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int32',
                                        name='edge_input')

        sentence_vector = sibiling_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(sibiling_model)(edge_input)

        main_output = keras.layers.Merge(mode=lambda i: K.batch_dot(i[0], i[1], axes=(1, 2)),
                                         name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, edge_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = input_to_indices.encode_batch_by_character(input_set, self._character2idx,
                                                                                    wdaccess.property2label,
                                                                                    max_input_len=self._p.get('max.sent.len', 70),
                                                                                    edge_with_entity=self._p.get('edge.with.entity', False))
        targets_as_one_hot = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets_as_one_hot

    def load_from_file(self, path_to_model):
        super(CharCNNModel, self).load_from_file(self, path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: character2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "character2idx_{}.json".format(self._model_number)) as f:
            self._character2idx = json.load(f)
        self._p['vocab.size'] = len(self._character2idx)
        self.logger.debug("Vocabulary size: {}.".format(len(self._character2idx)))


class YihModel(TwinsModel):

    def __init__(self, **kwargs):
        self._trigram2idx = defaultdict(int)
        super(YihModel, self).__init__(**kwargs)

    def encode_data_for_training(self, data_with_targets):
        pass

    def _get_keras_model(self):
        pass

    def encode_data_instance(self, instance):
        pass

    def load_from_file(self, path_to_model):
        super(YihModel, self).load_from_file(self, path_to_model=path_to_model)

        self.logger.debug("Loading vocabulary from: character2idx_{}.json".format(self._model_number))
        with open(self._save_model_to + "character2idx_{}.json".format(self._model_number)) as f:
            self._character2idx = json.load(f)
        self._p['vocab.size'] = len(self._character2idx)
        self.logger.debug("Vocabulary size: {}.".format(len(self._character2idx)))
