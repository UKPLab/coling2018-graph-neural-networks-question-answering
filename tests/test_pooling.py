import pytest
import yaml
import logging
import sys

from memory_profiler import profile

import models.pytorchmodel_impl

sys.path.append("../questionanswering")

from datasets import webquestions_io
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
config['model']['graph.choices'] = config['webquestions'].get("max.negative.samples", 30)
config['model']['epochs'] = 6
config['model']['threshold'] = 10
webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)

config['model']["batch.size"] = 10


def test_pool_selfatt():
    config['model']['sibling.pool.mode'] = "selfatt"
    config['model']["model.checkpoint"] = True
    trainablemodel = models.pytorchmodel_impl.CNNLabelsHashesModel(parameters=config['model'], logger=logger)
    trainablemodel.prepare_model(webquestions)
    trainablemodel.train(webquestions, validation_with_targets=webquestions.get_validation_samples())
    print('Training finished')
    trainablemodel.load_last_saved()
    print("Loaded successfully")


def test_pool_questiondot():
    config['model']['graph.mode'] = "first.edge.dot"
    # config['model']['twin.similarity'] = "dot"
    config['model']["model.checkpoint"] = True
    trainablemodel = models.pytorchmodel_impl.CNNLabelsHashesModel(parameters=config['model'], logger=logger)
    trainablemodel.prepare_model(webquestions)
    trainablemodel.train(webquestions, validation_with_targets=webquestions.get_validation_samples())
    print('Training finished')
    trainablemodel.load_last_saved()
    print("Loaded successfully")


if __name__ == '__main__':
    # pytest.main([__file__])
    test_pool_questiondot()
