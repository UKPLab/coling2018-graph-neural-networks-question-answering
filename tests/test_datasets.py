import pytest
import yaml
import numpy as np

from questionanswering.datasets import webquestions_io

with open("../questionanswering/default_config.yaml", 'r') as config_file:
    config = yaml.load(config_file.read())
webquestions = webquestions_io.WebQuestions(config['webquestions'])


def test_load_webquestions():
    assert len(webquestions.get_validation_with_gold()) == 2
    assert len(webquestions.get_validation_with_gold()[0]) == 1133


def test_access_sample():
    input_set, targets = webquestions.get_training_samples()
    assert len(input_set) == len(targets)
    assert type(targets[0]) == np.int32
    assert all(['edgeSet' in g for g in input_set[0]])


if __name__ == '__main__':
    pytest.main([__file__])
