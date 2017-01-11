import click
import utils
import yaml
import numpy as np
import logging
import sys

import models
from wikidata import wdaccess
from datasets import webquestions_io


@click.command()
@click.argument('path_to_model')
@click.argument('config_file_path', default="default_config.yaml")
def test_model(path_to_model, config_file_path):
    """

    :param path_to_model:
    :param config_file_path:
    :return:
    """
    config = utils.load_config(config_file_path)
    config_global = config.get('global', {})
    np.random.seed(config_global.get('random_seed', 1))

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)

    wdaccess.wdaccess_p['relation_qualifiers'] = config['wikidata'].get('qualifiers', False)

    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    trainablemodel = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    trainablemodel.load_from_file(path_to_model)

    print("Testing the model on silver data.")
    accuracy_on_silver = trainablemodel.test_on_silver(webquestions.get_validation_samples())
    print("Accuracy on silver data: {}".format(accuracy_on_silver))

    if config['wikidata'].get('evaluate', False):
        print("Testing the model on gold answers.")
        validation_graph_lists, validation_gold_answers = webquestions.get_validation_with_gold()
        print("Evaluate on {} validation questions.".format(len(validation_gold_answers)))
        successes, avg_metrics = trainablemodel.test((validation_graph_lists, validation_gold_answers), verbose=True)
        print("Successful predictions: {} ({})".format(len(successes), len(successes) / len(validation_gold_answers)))
        print("Average f1: {:.4},{:.4},{:.4}".format(*avg_metrics))


if __name__ == "__main__":
    test_model()
