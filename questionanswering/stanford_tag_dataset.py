import nltk
import json
import logging

logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    data_folder = "../data/"

    with open(data_folder + "input/webquestions.examples.test.json") as f:
        webquestions = json.load(f)
    logging.debug('Loaded WebQuestions, size: {}'.format(len(webquestions)))

    tokenizer = nltk.tokenize.stanford.StanfordTokenizer(path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/models-3.7.0/edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz", path_to_jar = "../resources/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    pos_tagger = nltk.tag.stanford.StanfordPOSTagger("models-3.7.0/edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger", path_to_jar = "../resources/stanford-postagger-full-2015-12-09/stanford-postagger-3.6.0.jar")
    webquestions_utterances_tokens = tokenizer.tokenize_sents([q_obj['utterance'] for q_obj in webquestions])
    logging.debug('Tokenized')
    webquestions_utterances_nes = ne_tagger.tag_sents(webquestions_utterances_tokens)
    logging.debug('NE tags')
    webquestions_utterances_poss = pos_tagger.tag_sents(webquestions_utterances_tokens)
    logging.debug('POS tags')

    webquestions_utterances_alltagged = [list(zip(tokens,
                                                  list(zip(*webquestions_utterances_nes[i]))[1],
                                                  list(zip(*webquestions_utterances_poss[i]))[1])) for i, tokens in enumerate(webquestions_utterances_tokens)]

    with open(data_folder + "generated/webquestions.examples.test.utterances.tagged.json", "w") as out:
        json.dump(webquestions_utterances_alltagged, out, indent=4)
    logger.debug("Saved")
