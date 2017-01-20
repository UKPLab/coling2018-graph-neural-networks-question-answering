import json
import logging
import sys

import click
import numpy as np
import tqdm
from construction import staged_generation
from datasets import webquestions_io
import utils
from wikidata import wdaccess, entity_linking


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
    # logging.basicConfig(level=config['logger']['level'])

    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)
    entity_linking.entity_linking_p["max.entity.options"] = config['generation']["max.entity.options"]
    wdaccess.wdaccess_p["restrict.hopup"] = config['wikidata'].get("restrict.hopup", False)
    wdaccess.update_sparql_clauses()
    logger.debug("max.entity.options: {}".format(entity_linking.entity_linking_p["max.entity.options"]))

    logger.debug('Extracting entities.')
    webquestions_entities = webquestions.extract_question_entities()

    logger.debug('Generating choice graphs')
    choice_graphs_sets = []
    len_webquestion = webquestions.get_dataset_size()
    if 'take_first' in config['generation']:
        print("Taking the first {} questions.".format(config['generation']['take_first']))
        len_webquestion = config['generation']['take_first']
    for i in tqdm.trange(len_webquestion):
        ungrounded_graph = {'edgeSet': [],
                            'entities': webquestions_entities[i][:config['generation'].get("max.num.entities", 1)]}
        choice_graphs = staged_generation.generate_without_gold(ungrounded_graph)
        choice_graphs_sets.append(choice_graphs)
        if i % 200 == 0:
            logger.debug("Average number of choices per question so far: {}".format(
                np.mean([len(graphs) for graphs in choice_graphs_sets])))

    logger.debug('Generation is finished')
    with open(config['generation']["save.choice.to"], 'w') as out:
        json.dump(choice_graphs_sets, out, sort_keys=True, indent=4)

    logger.debug("Query cache: {}".format(len(wdaccess.query_cache)))
    logger.debug("Number of answers covered: {}".format(
        sum(1 for graphs in choice_graphs_sets if len(graphs) > 0) / len_webquestion ))
    logger.debug("Average number of choices per question: {}".format(
        np.mean([len(graphs) for graphs in choice_graphs_sets])))


if __name__ == "__main__":
    generate()
