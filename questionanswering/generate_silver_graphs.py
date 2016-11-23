import numpy as np
import logging
import json
import nltk
import tqdm

import graph
import staged_generation

np.random.seed(1)

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    import argparse
    logger = logging.getLogger(__name__)

    data_folder = "../data/"

    with open(data_folder + "webquestions.examples.train.json") as f:
        webquestions = json.load(f)
    webquestions = webquestions[:100]
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    tokenizer = nltk.tokenize.stanford.StanfordTokenizer(path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/stanford-ner-2015-12-09/classifiers/english.nowiki.3class.caseless.distsim.crf.ser.gz",
                                                    path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    logger.debug('Initialized StanfordNERTagger')
    webquestions_utterances_tagged = ne_tagger.tag_sents(tokenizer.tokenize_sents([q_obj['utterance'] for q_obj in webquestions]))

    webquestions_entities = [graph.extract_entities(utterance_t) for utterance_t in webquestions_utterances_tagged]
    logger.debug('Extracted entities.')

    silver_dataset = []
    for i in tqdm.trange(len(webquestions)):
        logger.info(graph.construct_graphs(webquestions[i]['utterance'], webquestions_entities[i]))
        ungrounded_graph = {'tokens': [t for t,_ in webquestions_utterances_tagged[i]],
                            'edgeSet': [],
                            'entities': webquestions_entities[i]}
        logger.info("Generating from: {}".format(ungrounded_graph))
        generated_graphs = staged_generation.generate_with_gold(ungrounded_graph, webquestions[i])
        silver_dataset.append(generated_graphs)

    logger.debug("Silver dataset size: {}".format(len(silver_dataset)))

    with open(data_folder + "webquestions.examples.train.silvergraphs.json", 'w') as out:
        json.dump(silver_dataset, out, sort_keys=True, indent=4)

    print("Number of answers covered: {}".format(
        len([1 for graphs in silver_dataset if len(graphs) > 0 and any([g[1][2] > 0.0 for g in graphs])]) / len(webquestions) ))
    print("Average f1 of the silver data: {}".format(
        np.average([np.max([g[1][2] for g in graphs]) if len(graphs) > 0 else 0.0 for graphs in silver_dataset])))