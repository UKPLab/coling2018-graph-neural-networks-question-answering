import pytest
import yaml

from questionanswering.datasets import webquestions_io


def test_load_webquestions():
    with open("../questionanswering/default_config.yaml", 'r') as config_file:
        config = yaml.load(config_file.read())
    webquestions = webquestions_io.WebQuestions(config['webquestions'])
    assert len(webquestions.get_validation_with_gold()) == 2
    assert len(webquestions.get_validation_with_gold()[0]) == 1133
    assert all(g['edgeSet'] for g in webquestions.get_training_samples()[0][0])


if __name__ == '__main__':
    pytest.main([__file__])
