from collections import defaultdict
import keras
import numpy as np
import abc

from . import TrainableQAModel
from . import input_to_indices
from wikidata import wdaccess


class KerasModel(TrainableQAModel):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters, **kwargs):
        self._p = parameters
        self._save_model_to = self._p['models.save.path']
        super(KerasModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def _get_keras_model(self):
        """

        :return:
        """

    def train(self, data_with_targets, validation_with_targets=None):
        encoded_for_training = self.encode_data_for_training(data_with_targets)
        input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss" if validation_with_targets else "loss", patience=5, verbose=1),
            keras.callbacks.ModelCheckpoint(self._save_model_to + "{}.kerasmodel".format(self.__class__.__name__), save_best_only=True)
        ]
        self._p['graph_choices'] = 30
        self._model, self._sibling_model = self._get_keras_model(self._p)
        if validation_with_targets:
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=200, batch_size=128, verbose=1,
                                               validation_data=(encoded_validation[:-1], encoded_validation[-1]),
                                               callbacks=callbacks)
        else:
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=200, batch_size=128, verbose=1, callbacks=callbacks)
        self.logger.debug("Model training is finished.")


class CharCNNModel(TrainableQAModel):
    def __init__(self, parameters, **kwargs):
        self._p = parameters
        self._model, self._sibling_model = None, None
        self._character2idx = defaultdict(int)
        self._save_model_to = self._p['models.save.path']
        super(CharCNNModel, self).__init__(**kwargs)

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sibling_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._sibling_model.predict_on_batch(edges_encoded[0])
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))

        return np.argsort(predictions)[::-1]

    def encode_data_instance(self, instance, **kwargs):
        sentence_ids, edges_ids = input_to_indices.encode_by_character(instance, self._character2idx, wdaccess.property2label)
        sentence_ids = np.asarray(sentence_ids[:self._p.get('max.sent.len', 70)], dtype="int32")
        edges_ids = np.asarray(edges_ids, dtype="int32")
        edges_ids = edges_ids[:,:self._p.get('max.sent.len', 70)]
        return sentence_ids, edges_ids

    def train(self, data_with_targets, validation_with_targets=None):
        self.logger.debug('Training process started.')
        if not self._character2idx:
            self._character2idx = input_to_indices.get_character_index(
                [" ".join(graphs[0]['tokens']) for graphs in data_with_targets[0] if graphs])
            self.logger.debug('Character index created, size: {}'.format(len(self._character2idx)))

        encoded_for_training = self.encode_data_for_training(data_with_targets)
        input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
        self.logger.debug('Data encoded for training.')
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss" if validation_with_targets else "loss", patience=5, verbose=1),
            keras.callbacks.ModelCheckpoint(self._save_model_to + "{}.kerasmodel".format(self.__class__.__name__), save_best_only=True)
        ]
        self.logger.debug("Callback initialized. Save models to: {}.kerasmodel".format(self._save_model_to + self.__class__.__name__))
        self._p['vocab.size'] = len(self._character2idx)
        self._p['graph.choices'] = 30
        self._model, self._sibling_model = self._get_keras_model()
        if validation_with_targets:
            self.logger.debug("Start training with a validation sample.")
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            callback_history = self._model.fit(list(input_set), targets,
                                           nb_epoch=200, batch_size=128, verbose=1,
                                           validation_data=(encoded_validation[:-1], encoded_validation[-1]),
                                           callbacks=callbacks)
        else:
            self.logger.debug("Start training without a validation sample.")
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=200, batch_size=128, verbose=1, callbacks=callbacks)
        self.logger.debug("Model training is finished.")

    def _get_keras_model(self):
        self.logger.debug("Create keras model.")
        characters_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        character_embeddings = keras.layers.Embedding(output_dim=self._p['emb.dim'], input_dim=self._p['vocab.size'],
                                                      input_length=self._p['max.sent.len'],
                                                      mask_zero=False)(characters_input)
        self.logger.debug("Embeddings layer initialized.")
        sentence_vector = keras.layers.Convolution1D(self._p['conv.size'], self._p['conv.width'], border_mode='same')(character_embeddings)
        self.logger.debug("Convolution layer is initialized")
        sentence_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)
        self.logger.debug("Global max pooling layer is initialized")
        # semantic_vector = keras.layers.Dense(self._p['sem.layer.size'] * 3, activation='tanh')(sentence_vector)
        semantic_vector = keras.layers.Dense(self._p['sem.layer.size'], activation='tanh', name='semantic_vector')(sentence_vector)
        semantic_vector = keras.layers.Dropout(self._p['dropout.sibling'])(semantic_vector)
        self.logger.debug("Sibling model structure is finished")
        sibiling_model = keras.models.Model(input=[characters_input], output=[semantic_vector])
        self.logger.debug("Sibling model is finished.")
        sentence_input = keras.layers.Input(shape=(self._p['max.sent.len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(self._p['graph.choices'], self._p['max.sent.len'],), dtype='int32',
                                        name='edge_input')

        sentence_vector = sibiling_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(sibiling_model)(edge_input)

        main_output = keras.layers.Merge(mode=lambda i: keras.backend.batch_dot(i[0], i[1], axes=(1, 2)),
                                         name="edge_scores", output_shape=(self._p['graph.choices'],))([sentence_vector, edge_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        self.logger.debug("Model structured is finished")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.logger.debug("Model is compiled")
        return model, sibiling_model

    def encode_data_for_training(self, data_with_targets):
        input_set, targets = data_with_targets
        sentences_matrix, edges_matrix = input_to_indices.encode_batch_by_character(input_set, self._character2idx,
                                                                                    wdaccess.property2label,
                                                                                    max_input_len=self._p.get('max.sent.len', 70))
        targets_as_one_hot = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets_as_one_hot


class YihModel(TrainableQAModel):

    def __init__(self, parameters, **kwargs):
        self._p = parameters
        self._save_model_to = self._p['model.save.path']

        self._model, self._sibling_model = None, None

        self._trigram2idx = defaultdict(int)
        super(YihModel, self).__init__(**kwargs)

