from collections import defaultdict
import keras
from keras import backend as K
from keras.preprocessing import sequence
import numpy as np
import abc
import os
import json

from . import TrainableQAModel
from . import input_to_indices
from wikidata import wdaccess


class KerasModel(TrainableQAModel):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters, **kwargs):
        self._p = parameters
        self._save_model_to = self._p['models.save.path']
        self._model = None
        super(KerasModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def _get_keras_model(self):
        """
        Initialize tha structure of a Keras model and return an instance of a model to train. It may store other models
        as instance variables.
        :return: a Keras model to be trained
        """

    @abc.abstractmethod
    def load_from_file(self, path_to_model):
        """
        Load a Keras model from file.

        :param path_to_model: path to the model file.
        """

    def train(self, data_with_targets, validation_with_targets=None):
        self.logger.debug('Training process started.')

        encoded_for_training = self.encode_data_for_training(data_with_targets)
        input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
        self.logger.debug('Data encoded for training.')

        model_number = 0
        model_file_name = "{}_{}.kerasmodel".format(self.__class__.__name__, model_number)
        while os.path.exists(self._save_model_to + model_file_name):
            model_number += 1
            model_file_name = "{}_{}.kerasmodel".format(self.__class__.__name__, model_number)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss" if validation_with_targets else "loss", patience=5, verbose=1),
            keras.callbacks.ModelCheckpoint(self._save_model_to + model_file_name,
                                            monitor="val_loss" if validation_with_targets else "loss", save_best_only=True)
        ]
        self.logger.debug("Callbacks are initialized. Save models to: {}{}.kerasmodel".format(self._save_model_to, model_file_name))

        self._p['graph.choices'] = 30
        self.logger.debug(self._p)
        self._model = self._get_keras_model()
        if validation_with_targets:
            self.logger.debug("Start training with a validation sample.")
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=self._p.get("epochs", 200), batch_size=self._p.get("batch.size", 128),
                                               verbose=1,
                                               validation_data=(list(encoded_validation[:-1]), encoded_validation[-1]),
                                               callbacks=callbacks)
        else:
            self.logger.debug("Start training without a validation sample.")
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=self._p.get("epochs", 200), batch_size=self._p.get("batch.size", 128),
                                               verbose=1, callbacks=callbacks)
        self.logger.debug("Model training is finished.")


class CharCNNModel(KerasModel):
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
        sentence_ids, edges_ids = input_to_indices.encode_by_character(instance, self._character2idx, wdaccess.property2label)
        sentence_ids = sequence.pad_sequences([sentence_ids], maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int32")
        edges_ids = sequence.pad_sequences(edges_ids, maxlen=self._p.get('max.sent.len', 70), padding='post', truncating='post', dtype="int32")
        return sentence_ids, edges_ids

    def train(self, data_with_targets, validation_with_targets=None):
        if not self._character2idx:
            self._character2idx = input_to_indices.get_character_index(
                [" ".join(graphs[0]['tokens']) for graphs in data_with_targets[0] if graphs])
            self.logger.debug('Character index created, size: {}'.format(len(self._character2idx)))
            with open(self._save_model_to + "character2idx.json", 'w') as out:
                json.dump(self._character2idx, out, indent=2)
        self._p['vocab.size'] = len(self._character2idx)
        KerasModel.train(self, data_with_targets, validation_with_targets)

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
        sibiling_model = keras.models.Model(input=[characters_input], output=[semantic_vector], name='sibiling_model')
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

        # self._sibling_model = sibiling_model
        return model

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = input_to_indices.encode_batch_by_character(input_set, self._character2idx,
                                                                                    wdaccess.property2label,
                                                                                    max_input_len=self._p.get('max.sent.len', 70))
        targets_as_one_hot = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets_as_one_hot

    def load_from_file(self, path_to_model):
        self.logger.debug("Loading model from file.")
        self._model = keras.models.load_model(path_to_model)
        self._sibling_model = self._model.get_layer(name="sibiling_model")
        self.logger.debug("Sibling model: {}".format(self._sibling_model))
        with open(self._save_model_to + "character2idx.json") as f:
            self._character2idx = json.load(f)
        self._p['vocab.size'] = len(self._character2idx)
        self.logger.debug("Vocabualry size: {}.".format(len(self._character2idx)))
        self.logger.debug("Loaded successfully.")


class YihModel(KerasModel):

    def __init__(self, **kwargs):
        self._sibling_model = None
        self._trigram2idx = defaultdict(int)
        super(YihModel, self).__init__(**kwargs)

    def encode_data_for_training(self, data_with_targets):
        pass

    def _get_keras_model(self):
        pass

    def apply_on_instance(self, instance):
        pass

    def encode_data_instance(self, instance):
        pass


