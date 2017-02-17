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
    staged_generation.generation_p["use.whitelist"] = config['evaluation'].get("use.whitelist", False)
    logger.debug("max.entity.options: {}".format(entity_linking.entity_linking_p["max.entity.options"]))
    if 'hop.types' in config['wikidata']:
        stages.HOP_TYPES = set(config['wikidata']['hop.types'])
    if 'arg.types' in config['wikidata']:
        stages.ARG_TYPES = set(config['wikidata']['arg.types'])
    if 'filter.out.relation.classes' in config['wikidata']:
        wdaccess.FILTER_RELATION_CLASSES = set(config['wikidata']['filter.out.relation.classes'])

    with open(config['evaluation']['questions']) as f:
        webquestions_questions = json.load(f)
    webquestions = webquestions_io.WebQuestions(config['webquestions'], logger=logger)

    logger.debug('Extracting entities.')
    webquestions_entities = webquestions.extract_question_entities()

    logger.debug('Loading the model from: {}'.format(path_to_model))
    qa_model = getattr(models, config['model']['class'])(parameters=config['model'], logger=logger)
    qa_model.load_from_file(path_to_model)

    logger.debug('Testing')
    global_answers = []
    avg_metrics = np.zeros(3)
    len_webquestion = webquestions.get_dataset_size()
    for i in tqdm.trange(len_webquestion, ncols=100, ascii=True):
        question_entities = webquestions_entities[i]
        nes = [e for e in question_entities if e[1] != "NN"]
        if config['evaluation'].get('only.named.entities', False) and len(nes) > 0:
            question_entities = nes
        ungrounded_graph = {'tokens': webquestions.get_question_tokens(i),
                            'edgeSet': [],
                            'entities': question_entities[:config['evaluation'].get("max.num.entities", 1)]}
        chosen_graphs = staged_generation.generate_with_model(ungrounded_graph, qa_model, beam_size=config['evaluation'].get("beam.size", 10))
        model_answers = []
        g = ({},)
        if chosen_graphs:
            j = 0
            while not model_answers and j < len(chosen_graphs):
                g = chosen_graphs[j]
                model_answers = wdaccess.query_graph_denotations(g[0])
                j += 1
        if config['evaluation'].get('label.answers', False):
            gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(webquestions_questions[i])]
            model_answers_labels = wdaccess.label_query_results(model_answers)
            model_answers_labels = staged_generation.post_process_answers_given_graph(model_answers_labels, g[0])
            metrics = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers, model_answers_labels)
            global_answers.append((i, list(metrics), model_answers, model_answers_labels,
                                   [(c_g[0], float(c_g[1])) for c_g in chosen_graphs[:10]]))
        else:
            gold_answers = webquestions_io.get_answers_from_question(webquestions_questions[i])
            model_answers = [r["e1"] for r in model_answers]
            metrics = evaluation.retrieval_prec_rec_f1(gold_answers, model_answers)
            global_answers.append((i, list(metrics), model_answers,
                                   [(c_g[0], float(c_g[1])) for c_g in chosen_graphs[:10]]))
        avg_metrics += metrics
        if i % 100 == 0:
            logger.debug("Average f1 so far: {}".format((avg_metrics/(i+1))))
            with open(config['evaluation']["save.answers.to"], 'w') as answers_out:
                json.dump(global_answers, answers_out, sort_keys=True, indent=4)

    print("Average metrics: {}".format((avg_metrics/(len_webquestion))))

    logger.debug('Testing is finished')
    with open(config['evaluation']["save.answers.to"], 'w') as answers_out:
        json.dump(global_answers, answers_out, sort_keys=True, indent=4)

if __name__ == "__main__":
    generate()
