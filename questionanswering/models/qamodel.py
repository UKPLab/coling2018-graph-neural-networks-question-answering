import abc
from collections import deque
import tqdm
import numpy as np
import os
import re
import keras
from utils import Loggable

from wikidata import wdaccess
from datasets import evaluation


class QAModel(Loggable, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super(QAModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def encode_data_instance(self, instance):
        """
        Encode a single data instance in a format acceptable by the model.
        A data instance is a list of possible graphs.

        :param instance: a list of possible graphs for a single question.
        :return: a tuple that represents the instance in the model format.
        """

    def test(self, data_with_gold, verbose=False):
        graphs, gold_answers = data_with_gold
        predicted_indices = self.apply_on_batch(graphs, verbose=verbose)
        successes = deque()
        avg_metrics = np.zeros(3)
        for i, sorted_indices in enumerate(tqdm.tqdm(predicted_indices, ascii=True, disable=(not verbose))):
            sorted_indices = deque(sorted_indices)
            if sorted_indices:
                retrieved_answers = []
                while not retrieved_answers and sorted_indices:
                    index = sorted_indices.popleft()
                    g = graphs[i][index]
                    retrieved_answers = wdaccess.query_graph_denotations(g)
                retrieved_answers = wdaccess.map_query_results(retrieved_answers)
                metrics = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers[i], retrieved_answers)
                if metrics[-1]:
                    successes.append((i, metrics[-1], g))
                avg_metrics += metrics
        avg_metrics /= len(gold_answers)
        return successes, avg_metrics

    def test_on_silver(self, data_with_targets, verbose=False):
        graphs, targets = data_with_targets
        if len(targets) > 0 and not issubclass(type(targets[0]), np.integer):
            targets = np.argmax(targets, axis=-1)
        predicted_targets = [indices[0] for indices in self.apply_on_batch(graphs, verbose)]
        accuracy = np.sum(np.asarray(predicted_targets) == targets) / len(targets)
        return accuracy

    def apply_on_batch(self, data_batch, verbose=False):
        predicted_indices = deque()
        for instance in tqdm.tqdm(data_batch, ascii=True, disable=(not verbose)):
            predicted_indices.append(self.apply_on_instance(instance) if instance else [])
        return predicted_indices

    @abc.abstractmethod
    def apply_on_instance(self, instance):
        raise NotImplementedError


class TrainableQAModel(QAModel, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super(TrainableQAModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def train(self, data_with_targets, validation_with_targets=None):
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_for_training(self, data_with_targets):
        raise NotImplementedError


class KerasModel(TrainableQAModel, metaclass=abc.ABCMeta):
    def __init__(self, parameters, **kwargs):
        self._p = parameters
        self._save_model_to = self._p['models.save.path']
        self._model = None

        self._model_number = 0
        self._model_file_name = "{}_{}.kerasmodel".format(self.__class__.__name__, self._model_number)
        while os.path.exists(self._save_model_to + self._model_file_name):
            self._model_number += 1
            self._model_file_name = "{}_{}.kerasmodel".format(self.__class__.__name__, self._model_number)

        super(KerasModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def _get_keras_model(self):
        """
        Initialize tha structure of a Keras model and return an instance of a model to train. It may store other models
        as instance variables.
        :return: a Keras model to be trained
        """

    def load_from_file(self, path_to_model):
        """
        Load a Keras model from file.

        :param path_to_model: path to the model file.
        """
        self.logger.debug("Loading model from file.")
        self._model = keras.models.load_model(path_to_model)
        fname_match = re.search(r"_(\d+)\.", path_to_model)
        self._model_number = int(fname_match.group(1)) if fname_match else 0
        self.logger.debug("Loaded successfully.")

    @abc.abstractmethod
    def prepare_model(self, train_tokens):
        """
        Method that should override to init objects and parameters that are needed for the model training.
        E.g. vocabulary index.

        :param train_tokens:
        """
        self.logger.debug(self._p)
        assert "graph.choices" in self._p

        self._model = self._get_keras_model()

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

        callbacks = self.init_callbacks(monitor_validation=validation_with_targets)

        if validation_with_targets:
            self.logger.debug("Start training with a validation sample.")
            encoded_validation = self.encode_data_for_training(validation_with_targets)
            self.logger.debug('Validation data encoded for training.')
            self.logger.debug('Validating on {} samples.'.format(len(encoded_validation[-1])))
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
    def __init__(self, **kwargs):
        self._sibling_model = None
        self._sibling_model_name = "sibiling_model"

        super(TwinsModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def prepare_model(self, train_tokens):
        super(TwinsModel, self).prepare_model(train_tokens)
        self._sibling_model = self._model.get_layer(name="sibiling_model")
        self.logger.debug("Sibling model: {}".format(self._sibling_model))

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._sibling_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._sibling_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))
        if self._p.get("twin.similarity", "dot") == "cos":
            denominator = np.sqrt(
                np.sum(sentence_embedding * sentence_embedding) * np.sum(edge_embeddings * edge_embeddings, axis=-1))
            denominator = np.maximum(denominator, keras.backend.common._EPSILON)
            predictions /= denominator

        return np.argsort(predictions)[::-1]

    def load_from_file(self, path_to_model):
        super(TwinsModel, self).load_from_file(path_to_model=path_to_model)

        self._sibling_model = self._model.get_layer(name=self._sibling_model_name)
        self.logger.debug("Sibling model: {}".format(self._sibling_model))


class BrothersModel(KerasModel, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._older_model = None
        self._younger_model = None
        self._older_model_name = "older_model"
        self._younger_model_name = "younger_model"

        super(BrothersModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def prepare_model(self, train_tokens):
        super(BrothersModel, self).prepare_model(train_tokens)
        self._older_model = self._model.get_layer(name=self._older_model_name)
        self._younger_model = self._model.get_layer(name=self._younger_model_name).layer
        self.logger.debug("Older model: {}".format(self._older_model))
        self.logger.debug("Younger model: {}".format(self._younger_model))

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded = self.encode_data_instance(instance)
        sentence_embedding = self._older_model.predict_on_batch(tokens_encoded)[0]
        edge_embeddings = self._younger_model.predict_on_batch(edges_encoded)
        predictions = np.dot(sentence_embedding, np.swapaxes(edge_embeddings, 0, 1))
        if self._p.get("twin.similarity", "dot") == "cos":
            denominator = np.sqrt(
                np.sum(sentence_embedding * sentence_embedding) * np.sum(edge_embeddings * edge_embeddings, axis=-1))
            denominator = np.maximum(denominator, keras.backend.common._EPSILON)
            predictions /= denominator

        return np.argsort(predictions)[::-1]

    def load_from_file(self, path_to_model):
        super(BrothersModel, self).load_from_file(path_to_model=path_to_model)

        self._older_model = self._model.get_layer(name=self._older_model_name)
        self._younger_model = self._model.get_layer(name=self._younger_model_name)
        self.logger.debug("Older model: {}".format(self._older_model))
        self.logger.debug("Younger model: {}".format(self._younger_model))
