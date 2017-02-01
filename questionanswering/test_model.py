import json

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

    results_logger = None
    if 'log.results' in config['training']:
        results_logger = logging.getLogger("results_logger")
        results_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(filename=config['training']['log.results'])
        fh.setLevel(logging.INFO)
        results_logger.addHandler(fh)
        results_logger.info(str(config))

    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    config['model']['samples.per.epoch'] = webquestions.get_train_sample_size()
    config['model']['graph.choices'] = config['webquestions'].get("max.negative.samples", 30)

    trainablemodel = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    trainablemodel.load_from_file(path_to_model)

    print("Testing the model on silver data.")
    if 'train_validation' in config['webquestions']['path.to.dataset']:
        silver_test_set, silver_test_targets = webquestions.get_full_validation()
    else:
        silver_test_set, silver_test_targets = webquestions.get_full_training()
    accuracy_on_silver, predicted_targets = trainablemodel.test_on_silver((silver_test_set, silver_test_targets), verbose=True)
    print("Accuracy on silver data: {}".format(accuracy_on_silver))
    with open(config['training']['log.results'].replace(".log", "_silver_predictions.log"), "w") as out:
        json.dump((silver_test_set, silver_test_targets), out)
    if results_logger:
        results_logger.info("Accuracy on silver data: {}".format(accuracy_on_silver))

    if config['wikidata'].get('evaluate', False):
        wdaccess.wdaccess_p['wikidata_url'] = config['wikidata'].get("backend", "http://knowledgebase:8890/sparql")
        wdaccess.wdaccess_p["restrict.hop"] = config['wikidata'].get("restrict.hop", False)
        wdaccess.wdaccess_p["timeout"] = config['wikidata'].get("timeout", 20)
        wdaccess.sparql_init()
        wdaccess.update_sparql_clauses()

        validation_graph_lists, validation_gold_answers = webquestions.get_validation_with_gold()
        print("Evaluate on {} validation questions.".format(len(validation_gold_answers)))
        successes, avg_metrics = trainablemodel.test((validation_graph_lists, validation_gold_answers), verbose=True)
        print("Successful predictions: {} ({})".format(len(successes), len(successes) / len(validation_gold_answers)))
        print("Average f1: {:.4},{:.4},{:.4}".format(*avg_metrics))
        if results_logger:
            results_logger.info("Successful predictions: {} ({})".format(len(successes), len(successes) / len(validation_gold_answers)))
            results_logger.info("Average prec, rec, f1: {:.4}, {:.4}, {:.4}".format(*avg_metrics))


if __name__ == "__main__":
    test_model()
