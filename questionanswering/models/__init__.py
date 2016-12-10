import abc
from .input_to_indices import *
from .yih_et_al import *


class QAModel:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def encode_data_instance(instance):
        """
        Encode a single data instance in a format acceptable by the model.
        A data instance is a list of possible graphs.

        :param instance: a list of possible graphs for a single question.
        :return: a tuple that represents the instance in the model format.
        """

    @abc.abstractmethod
    def test(self, data_with_targets):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_on_batch(self, data_batch):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_on_instance(self, instance):
        raise NotImplementedError


class TrainableQAModel(QAModel):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def encode_data_for_training(self, data_with_targets):
        raise NotImplementedError
