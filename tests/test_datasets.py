import pytest
import yaml
import numpy as np
import logging
import models
import sys

import models.pytorchmodel_impl

sys.path.append("../questionanswering")

from datasets import webquestions_io

with open("../questionanswering/default_config.yaml", 'r') as config_file:
    config = yaml.load(config_file.read())

logger = logging.getLogger(__name__)
logger.setLevel(config['logger']['level'])
ch = logging.StreamHandler()
ch.setLevel(config['logger']['level'])
logger.addHandler(ch)

config['webquestions']['extensions'] = []
config['webquestions']['max.entity.options'] = 1
webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)


def test_load_webquestions():
    assert len(webquestions.get_full_validation()) == 2
    assert len(webquestions._dataset_tagged) == 200


def test_load_wikipedia():
    wikipedia = webquestions_io.Wikipedia(config['wikipedia'], logger=logger)
    training_samples, targets = wikipedia.get_training_samples()
    print(training_samples[0], targets[0])
    training_samples, targets = webquestions.get_training_samples()
    print(training_samples[0], targets[0])


def test_load_simplequestions():
    simplequestion = webquestions_io.SimpleQuestions(config['simplequestions'], logger=logger)
    simplequestion._idx2property = list(webquestions.get_property_set())
    training_samples, targets = simplequestion.get_training_samples()
    print(training_samples[0], targets[0])
    training_samples, targets = webquestions.get_training_samples()
    print(training_samples[0], targets[0])


def test_access_sample():
    input_set, targets = webquestions.get_training_samples()
    assert len(input_set) == len(targets)
    if config['webquestions'].get('target.dist'):
        assert len(targets[0]) == config['webquestions'].get('max.negative.samples')
    else:
        assert type(targets[0]) == np.int32
    print(input_set[0])
    assert all('edgeSet' in g for g in input_set[0][1])
    assert all(len(g['edgeSet']) > 0 for s in input_set for g in s[1])


def test_access_sample_with_model():
    trainablemodel = models.pytorchmodel_impl.CNNLabelsTorchModel(parameters=config['model'], logger=logger)
    trainablemodel.prepare_model(webquestions.get_question_tokens_set(), webquestions.get_property_set())
    input_set, targets = webquestions.get_training_samples(model=trainablemodel)
    assert len(input_set) == len(targets)
    if config['webquestions'].get('target.dist'):
        assert len(targets[0]) == config['webquestions'].get('max.negative.samples')
    else:
        assert type(targets[0]) == np.int32
    assert all('edgeSet' in g for g in input_set[0][1])
    assert all(len(g['edgeSet']) > 0 for s in input_set for g in s[1])


def test_replace_tokens():
    assert webquestions.get_training_tokens()[0] == ['what', 'sea', 'does', 'the', '<e>', 'flow', 'into', '?']


if __name__ == '__main__':
    test_load_simplequestions()

