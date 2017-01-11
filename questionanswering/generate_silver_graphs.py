import json
import logging
import click
import numpy as np
from construction import staged_generation
import tqdm
import sys

import utils
from wikidata import wdaccess
from construction import graph
from datasets import webquestions_io


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
def generate(config_file_path):
    config = utils.load_config(config_file_path)
    if "generation" not in config:
        print("Generation parameters not in the config file!")
        sys.exit()
    config_global = config.get('global', {})
    np.random.seed(config_global.get('random.seed', 1))

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)
    logging.basicConfig(level=logging.ERROR)

    wdaccess.wdaccess_p['relation_qualifiers'] = config['wikidata'].get('qualifiers', False)

    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    logging.debug('Loaded WebQuestions, size: {}'.format(webquestions.get_dataset_size()))

    with open(config['generation']['questions']) as f:
        webquestions_questions = json.load(f)
    logging.debug('Loaded WebQuestions original training questions, size: {}'.format(len(webquestions_questions)))
    assert len(webquestions_questions) == webquestions.get_dataset_size()

    logger.debug('Extracting entities.')
    webquestions_entities = webquestions.extract_question_entities()
    webquestions_tokens = webquestions.get_question_tokens()

    silver_dataset = []
    len_webquestion = webquestions.get_dataset_size()
    if 'take_first' in config['generation']:
        print("Taking the first {} questions.", config['generation']['take_first'])
        len_webquestion = config['generation']['take_first']
    for i in tqdm.trange(len_webquestion):
        ungrounded_graph = {'tokens': webquestions_tokens[i],
                            'edgeSet': [],
                            'entities': [webquestions_io.get_main_entity_from_question(webquestions_questions[i])] + webquestions_entities[i]}
        logger.info("Generating from: {}".format(ungrounded_graph))
        gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(webquestions_questions[i])]
        generated_graphs = staged_generation.generate_with_gold(ungrounded_graph, gold_answers)
        silver_dataset.append(generated_graphs)

    logger.debug("Generation finished. Silver dataset size: {}".format(len(silver_dataset)))
    with open(config['generation']["save.silver.to"], 'w') as out:
        json.dump(silver_dataset, out, sort_keys=True, indent=4)

    print("Query cache: {}".format(len(wdaccess.query_cache)))
    print("Number of answers covered: {}".format(
        len([1 for graphs in silver_dataset if len(graphs) > 0 and any([g[1][2] > 0.0 for g in graphs])]) / len_webquestion ))
    print("Average f1 of the silver data: {}".format(
        np.average([np.max([g[1][2] for g in graphs]) if len(graphs) > 0 else 0.0 for graphs in silver_dataset])))


if __name__ == "__main__":
    generate()