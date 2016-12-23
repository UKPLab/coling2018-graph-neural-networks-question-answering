import re
import json
import numpy as np
import itertools

from . import Dataset


class WebQuestions(Dataset):
    def __init__(self, path_to_dataset):
        # Load the train questions
        with open(path_to_dataset["webquestions.examples.train.train"]) as f:
            self._questions_train = json.load(f)
        # Load the validation questions
        with open(path_to_dataset["webquestions.examples.train.validation"]) as f:
            self._questions_val = json.load(f)
        # Load the tagged version
        with open(path_to_dataset["webquestions.examples.train.tagged"]) as f:
            self._dataset_tagged = json.load(f)
        # Load the generated graphs
        with open(path_to_dataset["webquestions.examples.train.silvergraphs"]) as f:
            self._silver_graphs = json.load(f)
        # Load the choice graphs. Choice graphs are all graph derivable from each sentence.
        with open(path_to_dataset["webquestions.examples.train.choicegraphs"]) as f:
            self._choice_graphs = json.load(f)
            self._choice_graphs = [[g[0] for g in graph_set] for graph_set in self._choice_graphs]
        assert len(self._dataset_tagged) == len(self._choice_graphs) == len(self._silver_graphs)

    def _get_samples(self, questions):
        indices = [q_obj['index'] for q_obj in questions
                   if any(len(g) > 1 and g[1][2] > 0.5 for g in self._silver_graphs[q_obj['index']]) and self._choice_graphs[
                       q_obj['index']]]
        return self._get_indexed_samples(indices)

    def _get_indexed_samples(self, indices):
        graph_lists = []
        targets = []
        for index in indices:
            graph_list = self._silver_graphs[index]
            negative_pool = [n_g for n_g in self._choice_graphs[index]
                             if all(n_g.get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]
            negative_pool_size = 30 - len(graph_list)
            if negative_pool:
                graph_list += [(n_g,) for n_g in np.random.choice(negative_pool,
                                                                  negative_pool_size,
                                                                  replace=len(negative_pool) < negative_pool_size)]
            np.random.shuffle(graph_list)
            target = np.argmax([g[1][2] if len(g) > 1 else 0.0 for g in graph_list])
            graph_list = [el[0] for el in graph_list]
            graph_lists.append(graph_list)
            targets.append(target)
        return graph_lists, targets

    def get_training_samples(self):
        return self._get_samples(self._questions_train)

    def get_validation_samples(self):
        return self._get_samples(self._questions_val)

    def get_training_generator(self, questions, batch_size):
        indices = [q_obj['index'] for q_obj in questions
                   if any(len(g) > 1 and g[1][2] > 0.5 for g in self._silver_graphs[q_obj['index']]) and
                   self._choice_graphs[q_obj['index']]]
        for i in itertools.cycle(range(0, len(indices), batch_size)):
            batch_indices = indices[i:i + batch_size]
            yield self._get_indexed_samples(batch_indices)

    def get_validation_with_gold(self):
        graph_lists = []
        gold_answers = []
        for q_obj in self._questions_val:
            index = q_obj['index']
            graph_list = self._choice_graphs[index]
            gold_answer = [e.lower() for e in get_answers_from_question(q_obj)]
            graph_lists.append(graph_list)
            gold_answers.append(gold_answer)
        return graph_lists, gold_answers


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
