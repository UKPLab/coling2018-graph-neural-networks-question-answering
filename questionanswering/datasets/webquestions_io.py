import re
import json

from . import Dataset


class WebQuestions(Dataset):
    def __init__(self, path_to_dataset):
        # Load the train questions
        with open(path_to_dataset + "input/webquestions.examples.train.train.json") as f:
            self._questions_train = json.load(f)
        # Load the validation questions
        with open(path_to_dataset + "input/webquestions.examples.train.validation.json") as f:
            self._questions_val = json.load(f)
        # Load the tagged version
        with open(path_to_dataset + "webquestions.examples.train.utterances.tagged.json") as f:
            self._dataset_tagged = json.load(f)
        # Load the generated graphs
        with open(path_to_dataset + "webquestions.examples.train.silvergraphs.full_11_29.json") as f:
            self._silver_graphs = json.load(f)
        # Load the choice graphs. Choice graphs are all graph derivable from each sentence.
        with open(path_to_dataset + "webquestions.examples.train.silvergraphs.full_11_29.json") as f:
            self._choice_graphs = json.load(f)

    def get_trainig_samples(self):
        train_indices = [q_obj['index'] for q_obj in self._questions_train]
        # webquestions_silver_graphs_for_training = [(q_obj['index'], g)
        #                                            for q_obj in self
        #                                            for g in webquestions_silver_graphs[q_obj['index']] if g[1][2] > 0.5]


def get_answers_from_question(question_object):
    """
    Retrieve a list of answers from a question as encoded in the WebQuestions dataset.

    :param question_object: A question encoded as a Json object
    :return: A list of answers as strings
    >>> get_answers_from_question({"url": "http://www.freebase.com/view/en/natalie_portman", "targetValue": "(list (description \\"Padm\u00e9 Amidala\\"))", "utterance": "what character did natalie portman play in star wars?"})
    ['Padmé Amidala']
    >>> get_answers_from_question({"targetValue": "(list (description Abduction) (description Eclipse) (description \\"Valentine's Day\\") (description \\"New Moon\\"))"})
    ['Abduction', 'Eclipse', "Valentine's Day", 'New Moon']
    """
    return re.findall("\(description \"?(.*?)\"?\)", question_object.get('targetValue'))


def get_main_entity_from_question(question_object):
    """
    Retrieve the main Freebase entity linked in the url field

    :param question_object: A question encoded as a Json object
    :return: A list of answers as strings
    >>> get_main_entity_from_question({"url": "http://www.freebase.com/view/en/natalie_portman", "targetValue": "(list (description \\"Padm\u00e9 Amidala\\"))", "utterance": "what character did natalie portman play in star wars?"})
    (['Natalie', 'Portman'], 'URL')
    >>> get_main_entity_from_question({"targetValue": "(list (description Abduction) (description Eclipse) (description \\"Valentine's Day\\") (description \\"New Moon\\"))"})
    ()
    """
    url = question_object.get('url')
    if url:
        entity_tokens = url.replace("http://www.freebase.com/view/en/", "").split("_")
        return [w.title() for w in entity_tokens], 'URL'
    return ()

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())