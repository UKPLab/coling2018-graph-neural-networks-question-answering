import json
import sys

import click
import numpy as np
import tqdm

import fackel

from questionanswering import config_utils, _utils
from questionanswering.construction import sentence
from questionanswering.grounding import staged_generation, graph_queries
from questionanswering.datasets import evaluation
from questionanswering.datasets import webquestions_io
from questionanswering.models import vectorization as V

from questionanswering import models


@click.command()
@click.argument('path_to_model')
@click.argument('config_file_path', default="default_config.yaml")
def generate(path_to_model, config_file_path):

    config, logger = config_utils.load_config(config_file_path)
    if "evaluation" not in config:
        print("Evaluation parameters not in the config file!")
        sys.exit()

    with open(config['evaluation']['questions']) as f:
        webquestions_questions = json.load(f)

    entitylinker = None
    if 'entity.linking' in config:
        PATH_EL = "../../entity-linking/"
        sys.path.insert(0, PATH_EL)
        from entitylinking import core
        linking_config = config['entity.linking']
        logger.info("Load entity linker")
        entitylinker = getattr(core, linking_config['linker'])(logger=logger,
                                                           **linking_config['linker.options'], pos_tags=True)

    _, word2idx = V.extend_embeddings_with_special_tokens(
        *_utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.100d.txt")
    )
    V.WORD_2_IDX = word2idx

    model_type = path_to_model.split("/")[-1].split("_")[0]
    logger.info(f"Model type: {model_type}")

    logger.info('Loading the model from: {}'.format(path_to_model))

    dummy_net = getattr(models, model_type)()
    container = fackel.TorchContainer(
        torch_model=dummy_net,
        logger=logger
    )
    container.load_from_file(path_to_model)

    graph_queries.FREQ_THRESHOLD = config['evaluation'].get("min.relation.freq", 500)
    logger.debug('Testing')
    global_answers = []
    avg_metrics = np.zeros(3)
    data_iterator = tqdm.tqdm(webquestions_questions, ncols=100, ascii=True)
    for i, q_obj in enumerate(data_iterator):
        q = q_obj.get('utterance', q_obj.get('question'))
        q_index = q_obj['questionid']

        if entitylinker:
            sent = entitylinker.link_entities_in_raw_input(q, element_id=q_index)
            if "max.num.entities" in config['evaluation']:
                sent.entities = sent.entities[:config['evaluation']["max.num.entities"]]
            sent = sentence.Sentence(input_text=sent.input_text, tagged=sent.tagged, entities=sent.entities)
        else:
            tagged = _utils.get_tagged_from_server(q, caseless=q.islower())
            sent = sentence.Sentence(input_text=q, tagged=tagged, entities=q_obj['entities'])

        chosen_graphs = staged_generation.generate_with_model(sent,
                                                              container,
                                                              beam_size=config['evaluation'].get("beam.size", 10))
        model_answers = []
        g = ({},)
        if chosen_graphs:
            j = 0
            while not model_answers and j < len(chosen_graphs):
                g = chosen_graphs[j]
                model_answers = graph_queries.get_graph_denotations(g.graph)
                j += 1

        gold_answers = webquestions_io.get_answers_from_question(q_obj)
        metrics = evaluation.retrieval_prec_rec_f1(gold_answers, model_answers)
        global_answers.append((q_index, list(metrics), model_answers,
                               [(c_g.graph, float(c_g.scores[2])) for c_g in chosen_graphs[:10]]))
        avg_metrics += metrics
        precision, recall, f1 = tuple(avg_metrics/(i+1))
        data_iterator.set_postfix(prec=precision,
                                  rec=recall,
                                  f1=f1)

        if i > 0 and i % 100 == 0:
            with open(config['evaluation']["save.answers.to"], 'w') as answers_out:
                json.dump(global_answers, answers_out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)

    print("Average metrics: {}".format((avg_metrics/(len(webquestions_questions)))))

    logger.debug('Testing is finished')
    with open(config['evaluation']["save.answers.to"], 'w') as answers_out:
        json.dump(global_answers, answers_out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)


if __name__ == "__main__":
    generate()
