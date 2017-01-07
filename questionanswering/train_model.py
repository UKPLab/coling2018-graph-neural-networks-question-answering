import logging
import click
import numpy as np
import yaml
import sys

from datasets import webquestions_io
import models
from models.qamodel import KerasModel


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
def train(config_file_path):
    """

    :param config_file_path:
    :return:
    """
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file.read())
    print(config)
    config_global = config.get('global', {})
    if "webquestions" not in config:
        print("Dataset location not in the config file!")
        sys.exit()

    if "model" not in config and 'class' not in config['model']:
        print("Specify a model class in the config file!")
        sys.exit()

    np.random.seed(config_global.get('random.seed', 1))

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)

    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    config['model']['samples.per.epoch'] = webquestions.get_train_sample_size()
    config['model']['graph.choices'] = config['webquestions'].get("max.negative.samples", 30)

    trainablemodel = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    if isinstance(trainablemodel, KerasModel):
        trainablemodel.prepare_model(webquestions.get_dataset_tokens())
    if config_global.get('train.generator', False):
        trainablemodel.train_on_generator(webquestions.get_training_generator(config['model'].get("batch.size", 128)),
                                          validation_with_targets=webquestions.get_validation_samples())
    else:
        trainablemodel.train(webquestions.get_training_samples(),
                             validation_with_targets=webquestions.get_validation_samples())

    trainablemodel.test_on_silver(webquestions.get_validation_samples())

    if config.get("wikidata", False):
        validation_graph_lists, validation_gold_answers = webquestions.get_validation_with_gold()
        print("Evaluate on {} validation questions.".format(len(validation_gold_answers)))
        trainablemodel.test((validation_graph_lists, validation_gold_answers), verbose=True)


if __name__ == "__main__":
    train()
