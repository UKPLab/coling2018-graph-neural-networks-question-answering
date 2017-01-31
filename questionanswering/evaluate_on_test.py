import json
import logging
import sys

import click
import numpy as np
import tqdm
from construction import staged_generation, stages
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

    wdaccess.wdaccess_p["timeout"] = config['wikidata'].get("timeout", 20)
    wdaccess.wdaccess_p['wikidata_url'] = config['wikidata'].get("backend", "http://knowledgebase:8890/sparql")
    wdaccess.sparql_init()

    entity_linking.entity_linking_p["max.entity.options"] = config['evaluation']["max.entity.options"]
    wdaccess.wdaccess_p["restrict.hop"] = config['wikidata'].get("restrict.hop", False)
    wdaccess.update_sparql_clauses()
    staged_generation.generation_p["replace.entities"] = config['webquestions'].get("replace.entities", False)
    staged_generation.generation_p["use.whitelist"] = config['evaluation'].get("use.whitelist", False)
    logger.debug("max.entity.options: {}".format(entity_linking.entity_linking_p["max.entity.options"]))
    if 'hop.types' in config['wikidata']:
        stages.HOP_TYPES = config['wikidata']['hop.types']
    if 'arg.types' in config['wikidata']:
        stages.ARG_TYPES = config['wikidata']['arg.types']

    with open(config['evaluation']['questions']) as f:
        webquestions_questions = json.load(f)
    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)

    logger.debug('Extracting entities.')
    webquestions_entities = webquestions.extract_question_entities()
    webquestions_tokens = webquestions.get_question_tokens()

    logger.debug('Loading the model from: {}'.format(path_to_model))
    qa_model = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    qa_model.load_from_file(path_to_model)

    logger.debug('Testing')
    answers_out = open(config['evaluation']["save.answers.to"], 'w')
    global_answers = []
    avg_metrics = np.zeros(3)
    len_webquestion = webquestions.get_dataset_size()
    for i in tqdm.trange(len_webquestion, ncols=100, ascii=True):
        ungrounded_graph = {'tokens': webquestions_tokens[i],
                            'edgeSet': [],
                            'entities': webquestions_entities[i][:config['evaluation'].get("max.num.entities", 1)]}
        gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(webquestions_questions[i])]
        chosen_graphs = staged_generation.generate_with_model(ungrounded_graph, qa_model, beam_size=config['evaluation'].get("beam.size", 10))
        model_answers = []
        if chosen_graphs:
            j = 0
            while not model_answers and j < len(chosen_graphs):
                g = chosen_graphs[j]
                model_answers = wdaccess.query_graph_denotations(g[0])
                j += 1
        model_answers_labels = wdaccess.label_query_results(model_answers)
        metrics = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers, model_answers_labels)
        avg_metrics += metrics
        global_answers.append((i, list(metrics), model_answers, model_answers_labels,
                               [(c_g[0], float(c_g[1])) for c_g in chosen_graphs[:10]]))
        if i % 100 == 0:
            logger.debug("Average f1 so far: {}".format((avg_metrics/(i+1))))
            json.dump(global_answers, answers_out, sort_keys=True, indent=4)

    print("Average metrics: {}".format((avg_metrics/(len_webquestion))))

    logger.debug('Testing is finished')
    json.dump(global_answers, answers_out, sort_keys=True, indent=4)
    answers_out.close()

if __name__ == "__main__":
    generate()
