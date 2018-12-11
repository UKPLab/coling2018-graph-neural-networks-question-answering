import abc
import logging


class Loggable(metaclass=abc.ABCMeta):

    def __init__(self, logger=None, **kwargs):
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"
epsilon = 10e-8