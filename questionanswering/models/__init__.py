from .input_to_indices import *
from .yih_et_al import *


class QAModel:

    @staticmethod
    def encode_data_instance(instance):
        raise NotImplementedError

    def test(self, data_with_targets):
        raise NotImplementedError

    def apply_on_batch(self, data_batch):
        raise NotImplementedError

    def apply_on_instance(self, instance):
        raise NotImplementedError


class TrainableQAModel(QAModel):

    def train(self, data):
        raise NotImplementedError

    def encode_data_for_training(self, data_with_targets):
        raise NotImplementedError
