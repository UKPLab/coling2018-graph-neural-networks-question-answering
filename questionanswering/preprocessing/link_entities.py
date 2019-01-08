# You need the Entity Linking for Wikidata project to run this script.
# Use the provided data sets if you just want to reproduce the experiments from the paper.

import json
import sys
import cv2

import click
import tqdm

PATH_EL = "../entity-linking/"
sys.path.insert(0, PATH_EL)
from entitylinking import linker

from questionanswering import config_utils


@click.command()
@click.argument('save_to')
@click.argument('config_file_path', default="default_config.yaml")
def generate(save_to, config_file_path):

    config, logger = config_utils.load_config(config_file_path)
    if "generation" not in config:
        logger.error("Generation parameters not in the config file!")
        sys.exit()
    config_global = config.get('global', {})
    linking_config = config['entity.linking']

    with open(config['generation']['questions']) as f:
        webquestions_questions = json.load(f)
    logger.info('Loaded training questions, size: {}'.format(len(webquestions_questions)))

    logger.info("Load entity linker")
    entitylinker = getattr(linker, linking_config['linker'])(logger=logger, **linking_config['linker.options'])

    len_webquestion = len(webquestions_questions)
    start_with = 0

    data_iterator = tqdm.tqdm(range(start_with, len_webquestion), ncols=100)
    for i in data_iterator:
        q_obj = webquestions_questions[i]
        q = q_obj.get('utterance', q_obj.get('question'))
        q_index = q_obj['questionid']

        sent = entitylinker.link_entities_in_raw_input(q, element_id=q_index)
        q_obj['entities'] = sent.entities

    logger.debug("Entity linking is finished.")
    with open(save_to, 'w') as out:
        json.dump(webquestions_questions, out, sort_keys=True, indent=4)


if __name__ == "__main__":
    generate()
