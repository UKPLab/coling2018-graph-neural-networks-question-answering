import numpy as np
import logging
import json
import nltk
import tqdm

import graph
import staged_generation

np.random.seed(1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    import argparse

    data_folder = "../data/"

    with open(data_folder + "webquestions.examples.train.json") as f:
        webquestions = json.load(f)
    webquestions = webquestions[:5]
    logger.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/stanford-ner-2015-12-09/classifiers/english.nowiki.3class.caseless.distsim.crf.ser.gz",
                                                    path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    logger.debug('Initialized StanfordNERTagger')

    webquestions_utterances_tagged = ne_tagger.tag_sents([q_obj['utterance'] for q_obj in webquestions])

    webquestions_entities = [graph.extract_entities(utterance_t) for utterance_t in webquestions_utterances_tagged]
    logger.debug('Extracted entities.')

    silver_dataset = []
    for i in tqdm.tnrange(len(webquestions)):
        logger.debug(graph.construct_graphs(webquestions[i]['utterance'], webquestions_entities[i]))
        ungrounded_graph = {'tokens': [t for t,_ in webquestions_utterances_tagged[i]],
                            'edgeSet': [],
                            'entities': webquestions_entities[i]}
        logger.debug("Generating from: {}".format(ungrounded_graph))
        generated_graphs = staged_generation.generate_with_gold(ungrounded_graph, webquestions[i])
        silver_dataset.extend(generated_graphs)

    logger.debug("Silver dataset size: {}".format(len(silver_dataset)))

    with open(data_folder + "webquestions.examples.train.silvergraphs.json") as out:
        json.dump(silver_dataset, out)

