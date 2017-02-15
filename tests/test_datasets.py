import pytest
import yaml
import numpy as np
import logging
import sys
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
    assert len(webquestions.get_question_tokens()) == 3778


def test_access_sample():
    input_set, targets = webquestions.get_training_samples()
    assert len(input_set) == len(targets)
    if config['webquestions'].get('target.dist'):
        assert len(targets[0]) == config['webquestions'].get('max.negative.samples')
    else:
        assert type(targets[0]) == np.int32
    assert all('edgeSet' in g for g in input_set[0])
    assert all(len(g['edgeSet']) > 0 for s in input_set for g in s)


if __name__ == '__main__':
    pytest.main([__file__])
