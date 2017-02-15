import click
import nltk
import json
import logging

import utils

logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.argument('config_file_path', default="default_config.yaml")
def tag_dataset(config_file_path):

    config = utils.load_config(config_file_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    with open(config['generation']['questions']) as f:
        webquestions = json.load(f)
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    tokenizer = nltk.tokenize.stanford.StanfordTokenizer(path_to_jar= "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/models-3.7.0/edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz", path_to_jar="../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    pos_tagger = nltk.tag.stanford.StanfordPOSTagger("../resources/models-3.7.0/edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger", path_to_jar="../resources/stanford-postagger-full-2015-12-09/stanford-postagger-3.6.0.jar")
    webquestions_utterances_tokens = tokenizer.tokenize_sents([q_obj.get('utterance', q_obj['question']) for q_obj in webquestions])
    logging.debug('Tokenized')
    webquestions_utterances_nes = ne_tagger.tag_sents(webquestions_utterances_tokens)
    logging.debug('NE tags')
    webquestions_utterances_poss = pos_tagger.tag_sents(webquestions_utterances_tokens)
    logging.debug('POS tags')

    webquestions_utterances_alltagged = [list(zip(tokens,
                                                  list(zip(*webquestions_utterances_nes[i]))[1],
                                                  list(zip(*webquestions_utterances_poss[i]))[1])) for i, tokens in enumerate(webquestions_utterances_tokens)]

    with open(config['webquestions']['path.to.dataset']['train_tagged'], "w") as out:
        json.dump(webquestions_utterances_alltagged, out, indent=4)
    logger.debug("Saved")

if __name__ == "__main__":
    tag_dataset()

