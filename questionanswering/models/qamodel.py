import abc
import logging
from collections import deque
import tqdm
import numpy as np

from wikidata import wdaccess
from datasets import evaluation


class QAModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger=None, **kwargs):
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

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
        predicted_indices = self.apply_on_batch(graphs)
        successes = deque()
        avg_f1 = 0.0
        for i, sorted_indices in enumerate(tqdm.tqdm(predicted_indices, ascii=True, disable=(not verbose))):
            sorted_indices = deque(sorted_indices)
            if sorted_indices:
                retrieved_answers = []
                while not retrieved_answers and sorted_indices:
                    index = sorted_indices.popleft()
                    g = graphs[i][index]
                    retrieved_answers = wdaccess.query_graph_denotations(g)
                retrieved_answers = wdaccess.map_query_results(retrieved_answers)
                _, _, f1 = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers[i], retrieved_answers)
                if f1:
                    successes.append((i, f1, g))
                avg_f1 += f1
        avg_f1 /= len(gold_answers)
        print("Successful predictions: {} ({})".format(len(successes), len(successes)/len(gold_answers)))
        print("Average f1: {}".format(avg_f1))

    def test_on_silver(self, data_with_targets, **kwargs):
        graphs, targets = data_with_targets
        predicted_targets = [indices[0] for indices in self.apply_on_batch(graphs)]
        accuracy = np.sum(np.asarray(predicted_targets) == targets) / len(targets)
        print("Accuarcy on silver tagets: {}".format(accuracy))

    def apply_on_batch(self, data_batch):
        predicted_indices = deque()
        for instance in data_batch:
            predicted_indices.append(self.apply_on_instance(instance) if instance else [])
        return predicted_indices

    @abc.abstractmethod
    def apply_on_instance(self, instance):
        raise NotImplementedError


class TrainableQAModel(QAModel):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(TrainableQAModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def train(self, data_with_targets, validation_with_targets=None):
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_for_training(self, data_with_targets):
        raise NotImplementedError


class KerasModel(TrainableQAModel):
    __metaclass__ = abc.ABCMeta

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

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss" if validation_with_targets else "loss", patience=5, verbose=1),
            keras.callbacks.ModelCheckpoint(self._save_model_to + self._model_file_name,
                                            monitor="val_loss" if validation_with_targets else "loss", save_best_only=True)
        ]
        self.logger.debug("Callbacks are initialized. Save models to: {}{}.kerasmodel".format(self._save_model_to, self._model_file_name))

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
