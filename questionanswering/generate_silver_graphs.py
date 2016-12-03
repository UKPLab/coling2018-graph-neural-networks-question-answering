import numpy as np
import logging
import json
import tqdm

import graph
import staged_generation
import webquestions_io
import wikidata_access

np.random.seed(1)
# TODO: read logging settings from a file
logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    data_folder = "../data/"

    with open(data_folder + "webquestions.examples.train.json") as f:
        webquestions = json.load(f)
    webquestions = webquestions[:100]
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    with open(data_folder + "webquestions.examples.train.utterances.tagged.json") as f:
        webquestions_utterances_alltagged = json.load(f)
    logger.debug('Loaded preprocessed data.')
    webquestions_utterances_alltagged = webquestions_utterances_alltagged[:100]
    assert len(webquestions) == len(webquestions_utterances_alltagged)

    webquestions_entities = [graph.extract_entities(webquestions_utterances_alltagged[i]) for i in range(len(webquestions))]
    logger.debug('Extracted entities.')

    silver_dataset = []
    for i in tqdm.trange(len(webquestions)):
        ungrounded_graph = {'tokens': [w for w, _, _ in webquestions_utterances_alltagged[i]],
                            'edgeSet': [],
                            'entities': [webquestions_io.get_main_entity_from_question(webquestions[i])] + webquestions_entities[i]}
        logger.info("Generating from: {}".format(ungrounded_graph))
        gold_answers = [e.lower() for e in webquestions_io.get_answers_from_question(webquestions[i])]
        generated_graphs = staged_generation.generate_with_gold(ungrounded_graph, gold_answers)
        silver_dataset.append(generated_graphs)

    logger.debug("Silver dataset size: {}".format(len(silver_dataset)))

    with open(data_folder + "webquestions.examples.train.silvergraphs.json", 'w') as out:
        json.dump(silver_dataset, out, sort_keys=True, indent=4)

    print("Query cache: {}".format(len(wikidata_access.query_cache)))
    print("Number of answers covered: {}".format(
        len([1 for graphs in silver_dataset if len(graphs) > 0 and any([g[1][2] > 0.0 for g in graphs])]) / len(webquestions) ))
    print("Average f1 of the silver data: {}".format(
        np.average([np.max([g[1][2] for g in graphs]) if len(graphs) > 0 else 0.0 for graphs in silver_dataset])))