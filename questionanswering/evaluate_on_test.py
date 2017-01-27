import json
import logging
import sys

import click
import numpy as np
import tqdm
from construction import staged_generation
from datasets import evaluation
from datasets import webquestions_io
import utils
from wikidata import wdaccess, entity_linking
import models


@click.command()
@click.argument('path_to_model')
@click.argument('config_file_path', default="default_config.yaml")
def generate(path_to_model, config_file_path):

    config = utils.load_config(config_file_path)
    if "evaluation" not in config:
        print("Evaluation parameters not in the config file!")
        sys.exit()
    config_global = config.get('global', {})
    np.random.seed(config_global.get('random.seed', 1))

    logger = logging.getLogger(__name__)
    logger.setLevel(config['logger']['level'])
    ch = logging.StreamHandler()
    ch.setLevel(config['logger']['level'])
    logger.addHandler(ch)
    # logging.basicConfig(level=config['logger']['level'])

    with open(config['evaluation']['questions']) as f:
        webquestions_questions = json.load(f)
    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)

    wdaccess.wdaccess_p["timeout"] = config['wikidata'].get("timeout", 20)
    wdaccess.wdaccess_p['wikidata_url'] = config['wikidata'].get("backend", "http://knowledgebase:8890/sparql")
    wdaccess.sparql_init()

    entity_linking.entity_linking_p["max.entity.options"] = config['evaluation']["max.entity.options"]
    wdaccess.wdaccess_p["restrict.hop"] = config['wikidata'].get("restrict.hop", False)
    wdaccess.update_sparql_clauses()
    logger.debug("max.entity.options: {}".format(entity_linking.entity_linking_p["max.entity.options"]))

    logger.debug('Extracting entities.')
    webquestions_entities = webquestions.extract_question_entities()

    logger.debug('Loading the model from: {}'.format(path_to_model))
    trainablemodel = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    trainablemodel.load_from_file(path_to_model)

    logger.debug('Testing')
    answers_out = open(config['evaluation']["save.answers.to"], 'w')
    global_answers = []
    avg_metrics = np.zeros(3)
    len_webquestion = webquestions.get_dataset_size()
    for i in tqdm.trange(len_webquestion):
        ungrounded_graph = {'edgeSet': [],
                            'entities': webquestions_entities[i][:config['generation'].get("max.num.entities", 1)]}
        gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(webquestions_questions[i])]
        model_answers = staged_generation.generate_with_model(ungrounded_graph)
        model_answers_labels = wdaccess.label_query_results(model_answers)
        metrics = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers[i], model_answers_labels)
        avg_metrics += metrics
        global_answers.append((i, metrics, model_answers, model_answers_labels))
        if i % 200 == 0:
            logger.debug("Average f1 so far: {}".format((avg_metrics/(i+1))))
            json.dump(global_answers, answers_out, sort_keys=True, indent=4)

    print("Average metrics: {}".format((avg_metrics/(len_webquestion))))

    logger.debug('Tesing is finished')
    json.dump(global_answers, answers_out, sort_keys=True, indent=4)
    answers_out.close()

if __name__ == "__main__":
    generate()
