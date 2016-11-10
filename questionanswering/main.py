import numpy as np
import logging
import json
import nltk
import tqdm

import graph

np.random.seed(1)
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    import argparse

    data_folder = "../data/"

    with open(data_folder + "webquestions.examples.train.json") as f:
        webquestions = json.load(f)
    webquestions = webquestions[:5]
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/stanford-ner-2015-12-09/classifiers/english.nowiki.3class.caseless.distsim.crf.ser.gz",
                                                    path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    logging.debug('Initialized StanfordNERTagger')

    webquestions_utterances_tagged = ne_tagger.tag_sents([q_obj['utterance'] for q_obj in webquestions])

    webquestions_entities = [graph.extract_entities(utterance_t) for utterance_t in webquestions_utterances_tagged]
    print(webquestions_entities)
    logging.debug('Extracted entities.')

    for i in range(len(webquestions)):
        print(graph.construct_graphs(webquestions[i]['utterance'], webquestions_entities[i]))

    # webquestions_entities
    #
    # for q_obj in webquestions:

