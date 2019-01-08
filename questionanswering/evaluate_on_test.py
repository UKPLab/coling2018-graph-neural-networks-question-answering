import json
import sys
import os
from collections import Counter

import click
import numpy as np
import tqdm

import fackel
from wikidata import queries

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
@click.argument('seed', default=-1)
@click.argument('gpuid', default=-1)
@click.argument('experiment_tag', default="")
def generate(path_to_model, config_file_path, seed, gpuid, experiment_tag):
    config, logger = config_utils.load_config(config_file_path, gpuid=gpuid, seed=seed)
    if "evaluation" not in config:
        print("Evaluation parameters not in the config file!")
        sys.exit()

    # Get the data set name and load the data set as specified in the config file
    dataset_name = config['evaluation']['questions'].split("/")[-1].split(".")[0]
    logger.info(f"Dataset: {dataset_name}")
    with open(config['evaluation']['questions']) as f:
        webquestions_questions = json.load(f)

    # Load the entity linker if specified, otherwise the entity annotations in the data set will be used
    entitylinker = None
    if 'entity.linking' in config:
        PATH_EL = "../../entity-linking/"
        sys.path.insert(0, PATH_EL)
        from entitylinking import core
        linking_config = config['entity.linking']
        logger.info("Load entity linker")
        entitylinker = getattr(core, linking_config['linker'])(logger=logger,
                                                           **linking_config['linker.options'], pos_tags=True)

    # Load the GloVe word embeddings and embeddings for special tokens
    _, word2idx = V.extend_embeddings_with_special_tokens(
        *_utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.100d.txt")
    )
    # Set the global mapping for words to indices
    V.WORD_2_IDX = word2idx

    # Derive the model type and the full model name from the model file
    model_type = path_to_model.split("/")[-1].split("_")[0]
    model_name = path_to_model.split("/")[-1].replace(".pkl", "")
    logger.info(f"Model type: {model_type}")
    logger.info('Loading the model from: {}'.format(path_to_model))

    # Load the PyTorch model
    dummy_net = getattr(models, model_type)()
    container = fackel.TorchContainer(
        torch_model=dummy_net,
        logger=logger
    )
    container.load_from_file(path_to_model)
    model_gated = container._model._gnn.hp_gated if model_type == "GNNModel" else False

    # Load the freebase entity set that was used top restrict the answer space by the previous work if specified.
    freebase_entity_set = set()
    if config['evaluation'].get('entities.list', False):
        print(f"Using the Freebase entity list")
        freebase_entity_set = _utils.load_blacklist(_utils.RESOURCES_FOLDER + "freebase-entities.txt")

    # Compose a file name for the output file
    save_answer_to = config['evaluation']["save.answers.to"]
    if not save_answer_to.endswith(".json"):
        dir_name = config['evaluation']["save.answers.to"] + f"{dataset_name}/{model_type.lower()}/"
        save_answer_to = dir_name + f"{dataset_name}_predictions_{'g' if model_gated else ''}{model_name.lower()}.json"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    print(f"Save output to {save_answer_to}")

    # Init the variables to store the results
    logger.debug('Testing')
    graph_queries.FREQ_THRESHOLD = config['evaluation'].get("min.relation.freq", 500)
    global_answers = []
    avg_metrics = np.zeros(4)

    # Iterate over the questions in the dataset
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
        j = -1
        if chosen_graphs:
            j = 0
            valid_answer_set = False
            while not valid_answer_set and j < len(chosen_graphs):
                g = chosen_graphs[j]
                model_answers = graph_queries.get_graph_denotations(g.graph)
                if model_answers:
                    valid_answer_set = True
                    if freebase_entity_set:
                        labeled_answers = {l.lower() for _, labels in
                                           queries.get_labels_for_entities(model_answers).items() for l in labels}
                        valid_answer_set = len(labeled_answers & freebase_entity_set) > len(model_answers) - 1
                j += 1

        gold_answers = webquestions_io.get_answers_from_question(q_obj)
        metrics = evaluation.retrieval_prec_rec_f1(gold_answers, model_answers)
        global_answers.append((q_index, list(metrics), model_answers,
                               [(c_g.graph, float(c_g.scores[2])) for c_g in chosen_graphs[:10]]))
        avg_metrics += metrics + (j,)
        precision, recall, f1, g_j = tuple(avg_metrics/(i+1))
        data_iterator.set_postfix(prec=precision,
                                  rec=recall,
                                  f1=f1, g_j=g_j)

        # Save intermediate results
        if i > 0 and i % 100 == 0:
            with open(save_answer_to, 'w') as answers_out:
                json.dump(global_answers, answers_out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)

    avg_metrics = avg_metrics / (len(webquestions_questions))
    print("Average metrics: {}".format(avg_metrics))

    # Fine-grained results, if there is a mapping of questions to the number of relation to find the correct answer
    results_by_hops = {}
    if "qid2hop" in config['evaluation']:
        with open(config['evaluation']['qid2hop']) as f:
            q_index2hop = json.load(f)
        print("Results by hop: ")
        hops_dist = Counter([q_index2hop[p[0]] for p in global_answers])
        results_by_hops = {i: np.zeros(3) for i in range(max(hops_dist.keys()) + 1)}
        for p in global_answers:
            metrics = tuple(p[1])
            results_by_hops[q_index2hop[p[0]]] += metrics
        for m in results_by_hops:
            if hops_dist[m] > 0:
                results_by_hops[m] = results_by_hops[m] / hops_dist[m]
        print(results_by_hops)

    # Add results to the results file
    if "add.results.to" in config['evaluation']:
        print(f"Adding results to {config['evaluation']['add.results.to']}")
        with open(config['evaluation']["add.results.to"], 'a+') as results_out:
            results_out.write(",".join([model_name,
                                        model_type,
                                        "Gated" if model_gated else "Simple",
                                        str(seed),
                                        dataset_name,
                                        "full",
                                        "EntityList" if freebase_entity_set else "NoEntityList"]
                                       + [str(el) for el in avg_metrics[:3]])
                              )
            results_out.write("\n")
            # Include fine grained results if available
            if results_by_hops:
                for i in range(max(results_by_hops.keys()) + 1):
                    results_out.write(",".join([model_name,
                                                model_type,
                                                "Gated" if model_gated else "Simple",
                                                container.description,
                                                str(seed),
                                                dataset_name,
                                                str(i),
                                                "EntityList" if freebase_entity_set else "NoEntityList"]
                                               + [str(el) for el in results_by_hops[i]]
                                               + [experiment_tag])
                                      )
                    results_out.write("\n")

    # Save final model output
    with open(save_answer_to, 'w') as answers_out:
        json.dump(global_answers, answers_out, sort_keys=True, indent=4, cls=sentence.SentenceEncoder)


if __name__ == "__main__":
    generate()
