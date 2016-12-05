import logging
import click
import numpy as np
import yaml
import json
import models


@click.command()
@click.argument('config_file_path')
@click.option('data_folder', default="../data")
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

    with open(data_folder + "webquestions.examples.train.silvergraphs.full_11_29.json") as f:
        webquestions_silver_graphs = json.load(f)

    with open(data_folder + "properties-with-labels.txt") as infile:
        property2label = {l.split("\t")[0] : l.split("\t")[1].strip() for l in infile.readlines()}

    max_sent_len = config['experiments'].get('max_sent_len', 60)
    max_property_len = config['experiments'].get('max_property_len', 60)

    trigram2idx = models.get_trigram_index()
    logger.info("Trigram vocabulary size: {}".format(trigram2idx))

    sentences_matrix, edges_matrix = models.encode_by_trigrams(webquestions_silver_graphs_for_training, trigram2idx, property2label)






if __name__ == "__main__":
    train()
