import abc

from questionanswering import base_objects


class Dataset(base_objects.Loggable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_training_samples(self, model=None):
        raise NotImplementedError
