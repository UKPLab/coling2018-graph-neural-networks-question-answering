import numpy as np
import logging
import json
import nltk

np.random.seed(1)
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    import argparse

    data_folder = "../data/"

    with open(data_folder + "webquestions.examples.train.json") as f:
        webquestions = json.load(f)
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/stanford-ner-2015-12-09/classifiers/english.nowiki.3class.caseless.distsim.crf.ser.gz",
                                                    path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    logging.debug('Initialized StanfordNERTagger')

