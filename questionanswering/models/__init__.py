from .input_to_indices import *
from .yih_et_al import *


class QAModel:

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def encode_data(self, data):
        raise NotImplementedError