from sklearn import linear_model

import pytest
import yaml
import numpy as np
import logging

from questionanswering.datasets import webquestions_io
import models

with open("../questionanswering/default_config.yaml", 'r') as config_file:
    config = yaml.load(config_file.read())

logger = logging.getLogger(__name__)
logger.setLevel(config['logger']['level'])
ch = logging.StreamHandler()
ch.setLevel(config['logger']['level'])
logger.addHandler(ch)

config['webquestions']['extensions'] = []
config['webquestions']['max.entity.options'] = 1
config['webquestions']['target.dist'] = False
del config['webquestions']['path.to.dataset']['train_validation']


def test_model_train():
    trainablemodel = models.BagOfWordsModel(parameters=config['model'], logger=logger)
    assert type(trainablemodel._model) == linear_model.LogisticRegression
    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    input_set, targets = webquestions.get_training_samples()
    input_set, targets = input_set[:200], targets[:200]
    trainablemodel.train((input_set, targets),
                         validation_with_targets=webquestions.get_validation_samples()
                         if 'train_validation' in config['webquestions']['path.to.dataset'] else None)
    print('Training finished')


if __name__ == '__main__':
    pytest.main([__file__])
