import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qamodel import TrainableQAModel
from models.inputbasemodel import TrigramBasedModel


class TorchModel(TrainableQAModel, metaclass=abc.ABCMeta):
    """
    This version of the model uses the Keras library for internal implementation.
    """
    def __init__(self, **kwargs):
        self._file_extension = "torchmodel"
        super(TorchModel, self).__init__(**kwargs)

    @abc.abstractmethod
    def prepare_model(self, train_tokens, properties_set):
        """
        Method that should override to init objects and parameters that are needed for the model training.
        E.g. vocabulary index.

        :param train_tokens:
        :param properties_set:
        """
        self.logger.debug(self._p)
        assert "graph.choices" in self._p
        assert "vocab.size" in self._p
        assert self._p["vocab.size"] > 0
        self._model = self._get_torch_net()

    @abc.abstractmethod
    def _get_torch_net(self):
        """
        Initialize tha structure of a Torch model and return an instance of a model to train. It may store other models
        as instance variables.
        :return: a Keras model to be trained
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_from_file(self, path_to_model):
        """
        Load a Keras model from file.

        :param path_to_model: path to the model file.
        """
        raise NotImplementedError

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

    # def init_callbacks(self, monitor_validation=False):
    #     monitor_value = ("val_" if monitor_validation else "") + self._p.get("monitor", "loss")
    #     callbacks = [
    #         keras.callbacks.EarlyStopping(monitor=monitor_value, patience=self._p.get("early.stopping", 5), verbose=1),
    #         keras.callbacks.ModelCheckpoint(self._save_model_to + self._model_file_name,
    #                                         monitor=monitor_value, save_best_only=True)
    #     ]
    #     self.logger.debug("Callbacks are initialized. Save models to: {}{}.kerasmodel".format(self._save_model_to,
    #                                                                                           self._model_file_name))
    #     return callbacks

    def data_for_training_generator(self, data_with_targets_generator):
        for data_with_targets in data_with_targets_generator:
            encoded_for_training = self.encode_data_for_training(data_with_targets)
            input_set, targets = encoded_for_training[:-1], encoded_for_training[-1]
            yield list(input_set), targets


class CNNLabelsTorchModel(TrigramBasedModel, TorchModel):
    def _get_torch_net(self):
        pass

    def scores_for_instance(self, instance):
        pass

    def encode_data_instance(self, instance):
        pass


class CNNLabelsTorchNet(nn.Module):
    def __init__(self):
        super(CNNLabelsTorchNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        F.
        return x