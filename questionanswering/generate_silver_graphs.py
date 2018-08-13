import json
import sys

import click
import numpy as np
import tqdm

PATH_EL = "../entity-linking/"
sys.path.insert(0, PATH_EL)
from entitylinking import core

from questionanswering import config_utils
from questionanswering.construction import graph, sentence
from questionanswering.grounding import staged_generation

from questionanswering.datasets import webquestions_io


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
def generate(config_file_path):

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
    entitylinker = getattr(core, linking_config['linker'])(logger=logger, **linking_config['linker.options'])

    silver_dataset = []
    previous_silver = []
    if 'previous' in config['generation']:
        logger.debug("Loading the previous result")
        with open(config['generation']['previous']) as f:
            previous_silver = json.load(f,  object_hook=sentence.sentence_object_hook)
            logger.info(f"Train: {len(previous_silver)}")
        print(f"Previous number of answers covered: "
              f"{len([1 for s in previous_silver if len(s.graphs) > 0 and any([g.scores[2] > 0.0 for g in s.graphs])]) / len(previous_silver)}")
        print(f"Previous average f1 of the silver data: "
              f"{np.average([np.max([g.scores[2] for g in s.graphs]) if len(s.graphs) > 0 else 0.0 for s in previous_silver])}")
        print(f"Reusable: "
              f"{len([1 for s in previous_silver if len(s.graphs) > 0 and any([g.scores[2] > 0.9 for g in s.graphs])]) / len(previous_silver)}")

    len_webquestion = len(webquestions_questions)
    start_with = 0
    if 'start.with' in config['generation']:
        start_with = config['generation']['start.with']
        print("Starting with {}.".format(start_with))

    data_iterator = tqdm.tqdm(range(start_with, len_webquestion), ncols=100)
    for i in data_iterator:
        if len(previous_silver) > i and previous_silver[i].graphs \
                and max(g.scores[2] for g in previous_silver[i].graphs) > 0.8:
            silver_dataset.append(previous_silver[i])
        else:
            q_obj = webquestions_questions[i]
            q = q_obj.get('utterance', q_obj.get('question'))
            q_index = q_obj['questionid']

            sent = entitylinker.link_entities_in_raw_input(q, element_id=q_index)
            if "max.num.entities" in config['generation']:
                sent.entities = sent.entities[:config['generation']["max.num.entities"]]

            sent = sentence.Sentence(input_text=sent.input_text, tagged=sent.tagged, entities=sent.entities)

            gold_answers = webquestions_io.get_answers_from_question(q_obj)
            if gold_answers and any(gold_answers):
                sent.graphs = staged_generation.generate_with_gold(sent.graphs[0], gold_answers)
            silver_dataset.append(sent)

        coverage = len([1 for s in silver_dataset if len(s.graphs) > 0 and any([g.scores[2] > 0.0 for g in s.graphs])]) / (i+1)
        avg_f1 = np.average([np.max([g.scores[2] for g in s.graphs]) if len(s.graphs) > 0 else 0.0 for s in silver_dataset])
        data_iterator.set_postfix(cov=coverage, f1=avg_f1)
        if i > 0 and i % 25 == 0:
            # Dump the data set once in while
            with open(config['generation']["save.silver.to"], 'w') as out:
                json.dump(silver_dataset, out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)

    logger.debug("Generation finished. Silver dataset size: {}".format(len(silver_dataset)))
    with open(config['generation']["save.silver.to"], 'w') as out:
        json.dump(silver_dataset, out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)

    print("Number of answers covered: {}".format(
        len([1 for s in silver_dataset if len(s.graphs) > 0 and any([g.scores[2] > 0.0 for g in s.graphs])]) / len_webquestion ))
    print("Average f1 of the silver data: {}".format(
        np.average([np.max([g.scores[2] for g in s.graphs]) if len(s.graphs) > 0 else 0.0 for s in silver_dataset])))


if __name__ == "__main__":
    generate()
