import abc
import keras
import numpy as np
import re
from models.qamodel import TrainableQAModel


class KerasModel(TrainableQAModel, metaclass=abc.ABCMeta):
    """
    This version of the model uses the Keras library for internal implementation.
    """
    def __init__(self, **kwargs):
        self._file_extension = "kerasmodel"
        super(KerasModel, self).__init__(**kwargs)

        self.logger.debug(self._p)
        assert "graph.choices" in self._p
        self._model = self._get_keras_model()

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
        super(KerasModel, self).load_from_file(path_to_model=path_to_model)
        self.logger.debug("Loading model from file.")
        self._model = keras.models.load_model(path_to_model)
        fname_match = re.search(r"_(\d+)\.", path_to_model)
        self._model_number = int(fname_match.group(1)) if fname_match else 0
        self.logger.debug("Loaded successfully.")

    def train(self, data_with_targets, validation_with_targets=None):
        self.logger.debug('Training process started.')

        encoded_for_training = self.encode_data_for_training(data_with_targets)
        input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
        self.logger.debug('Data encoded for training.')

        callbacks = self.init_callbacks(monitor_validation=validation_with_targets)

        if validation_with_targets:
            self.logger.debug("Start training with a validation sample.")
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            self.logger.debug('Validation data encoded for training.')
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=self._p.get("epochs", 200),
                                               batch_size=self._p.get("batch.size", 128),
                                               verbose=1,
                                               validation_data=(list(encoded_validation[:-1]), encoded_validation[-1]),
                                               callbacks=callbacks)
        else:
            self.logger.debug("Start training without a validation sample.")
            callback_history = self._model.fit(list(input_set), targets,
                                               nb_epoch=self._p.get("epochs", 200),
                                               batch_size=self._p.get("batch.size", 128),
                                               verbose=1, callbacks=callbacks)
        self.logger.debug("Model training is finished.")

    def train_on_generator(self, data_with_targets_generator, validation_with_targets=None):
        self.logger.debug('Training process with a generator started.')

        callbacks = self.init_callbacks(monitor_validation=validation_with_targets is not None)

        if validation_with_targets:
            self.logger.debug("Start training with a validation sample.")
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            self.logger.debug('Validation data encoded for training.')
            self.logger.debug('Validating on {} samples.'.format(len(encoded_validation[-1])))
            self.logger.debug('Validation shapes:{}'.format([s.shape for s in encoded_validation]))
            callback_history = self._model.fit_generator(self.data_for_training_generator(data_with_targets_generator),
                                                         nb_epoch=self._p.get("epochs", 200),
                                                         samples_per_epoch=self._p.get("samples.per.epoch", 1000),
                                                         verbose=1,
                                                         validation_data=(
                                                             list(encoded_validation[:-1]), encoded_validation[-1]),
                                                         callbacks=callbacks)
        else:
            self.logger.debug("Start training without a validation sample.")
            callback_history = self._model.fit_generator(self.data_for_training_generator(data_with_targets_generator),
                                                         nb_epoch=self._p.get("epochs", 200),
                                                         samples_per_epoch=self._p.get("samples.per.epoch", 1000),
                                                         verbose=1, callbacks=callbacks)
        self.logger.debug("Model training is finished.")

    def init_callbacks(self, monitor_validation=False):
        monitor_value = ("val_" if monitor_validation else "") + self._p.get("monitor", "loss")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor=monitor_value, patience=self._p.get("early.stopping", 5), verbose=1),
            keras.callbacks.ModelCheckpoint(self._save_model_to + self._model_file_name,
                                            monitor=monitor_value, save_best_only=True)
        ]
        self.logger.debug("Callbacks are initialized. Save models to: {}{}.kerasmodel".format(self._save_model_to,
                                                                                              self._model_file_name))
        return callbacks

    def data_for_training_generator(self, data_with_targets_generator):
        for data_with_targets in data_with_targets_generator:
            encoded_for_training = self.encode_data_for_training(data_with_targets)
            input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
            yield list(input_set), targets


class TwinsModel(KerasModel, metaclass=abc.ABCMeta):
    """
    TwinsModel is a Keras model that uses the same model to encode both the sentence and the candidate graphs.
    """
    def __init__(self, **kwargs):
        self._sibling_model = None
        self._sibling_model_name = "sibiling_model"

        super(TwinsModel, self).__init__(**kwargs)

        assert self._model is not None
        self._sibling_model = self._model.get_layer(name="sibiling_model")
        self.logger.debug("Sibling model: {}".format(self._sibling_model))

    def scores_for_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sibling_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._sibling_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))
        if self._p.get("twin.similarity", "dot") == "cos":
            denominator = np.sqrt(
                np.sum(sentence_embedding * sentence_embedding) * np.sum(edge_embeddings * edge_embeddings, axis=-1))
            denominator = np.maximum(denominator, keras.backend.common._EPSILON)
            predictions /= denominator
        return predictions

    def load_from_file(self, path_to_model):
        super(TwinsModel, self).load_from_file(path_to_model=path_to_model)

        self._sibling_model = self._model.get_layer(name=self._sibling_model_name)
        self.logger.debug("Sibling model: {}".format(self._sibling_model))


class BrothersModel(KerasModel, metaclass=abc.ABCMeta):
    """
    BrothersModel is a KerasModel that uses two distinct models to encode the sentence and the graphs.
    """
    def __init__(self, **kwargs):
        self._sentence_model = None
        self._graph_model = None
        self._sentence_model_name = "sentence_model"
        self._graph_model_name = "graph_model"

        super(BrothersModel, self).__init__(**kwargs)

        assert self._model is not None
        self._sentence_model = self._model.get_layer(name=self._sentence_model_name)
        self._graph_model = self._model.get_layer(name=self._graph_model_name).layer
        self.logger.debug("Sentence model: {}".format(self._sentence_model))
        self.logger.debug("Graph model: {}".format(self._graph_model))

    def scores_for_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sentence_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._graph_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))
        if self._p.get("twin.similarity", "dot") == "cos":
            denominator = np.sqrt(
                np.sum(sentence_embedding * sentence_embedding) * np.sum(edge_embeddings * edge_embeddings, axis=-1))
            denominator = np.maximum(denominator, keras.backend.common._EPSILON)
            predictions /= denominator
        return predictions

    def load_from_file(self, path_to_model):
        super(BrothersModel, self).load_from_file(path_to_model=path_to_model)

        self._sentence_model = self._model.get_layer(name=self._sentence_model_name)
        self._graph_model = self._model.get_layer(name=self._graph_model_name).layer
        self.logger.debug("Sentence model: {}".format(self._sentence_model))
        self.logger.debug("Graph model: {}".format(self._graph_model))