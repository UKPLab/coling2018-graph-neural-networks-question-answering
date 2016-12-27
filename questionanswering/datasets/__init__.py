import abc
import logging


class Dataset:
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger=None, **kwargs):
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
