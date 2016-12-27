import logging
import click
import numpy as np
import yaml

from datasets import webquestions_io
from models import baselines, char_based


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
@click.option('-data_folder', default="../data")
def train(config_file_path, data_folder):
    """

    :param config_file_path:
    :param data_folder:
    :return:
    """
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file.read())
    print(config)
    config_global = config.get('global', {})
    np.random.seed(config_global.get('random_seed', 1))

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)

    if "webquestions" not in config:
        print("Dataset location not in the config file!")
        exit()

    webquestions = webquestions_io.WebQuestions(config['webquestions'])

    # trainablemodel = baselines.BagOfWordsModel(config.get('model', {}), logger=logger)
    trainablemodel = char_based.CharCNNModel(config.get('model', {}), logger=logger)
    trainablemodel.train(webquestions.get_training_samples(), validation_with_targets=webquestions.get_validation_samples())

    trainablemodel.test_on_silver(webquestions.get_validation_samples())

    if config.get("wikidata", False):
        validation_graph_lists, validation_gold_answers = webquestions.get_validation_with_gold()
        print("Evaluate on {} validation questions.".format(len(validation_gold_answers)))
        trainablemodel.test((validation_graph_lists, validation_gold_answers), verbose=True)

if __name__ == "__main__":
    train()
