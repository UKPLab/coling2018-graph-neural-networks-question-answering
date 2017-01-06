import re
import json
import numpy as np
import itertools

from . import Dataset
from construction import graph


class WebQuestions(Dataset):
    def __init__(self, parameters, **kwargs):
        """
        An object class to access the webquestion dataset. The path to the dataset should point to a folder that
        contains a preprocessed dataset.

        :param path_to_dataset: path to the data set location
        """
        # TODO: Tests needed!
        self._p = parameters
        path_to_dataset = self._p["path.to.dataset"]
        # Load the train questions
        with open(path_to_dataset["train_train"]) as f:
            self._questions_train = json.load(f)
        # Load the validation questions
        with open(path_to_dataset["train_validation"]) as f:
            self._questions_val = json.load(f)
        # Load the tagged version
        with open(path_to_dataset["train_tagged"]) as f:
            self._dataset_tagged = json.load(f)
        # Load the generated graphs
        with open(path_to_dataset["train_silvergraphs"]) as f:
            self._silver_graphs = json.load(f)
        # Load the choice graphs. Choice graphs are all graph derivable from each sentence.
        with open(path_to_dataset["train_choicegraphs"]) as f:
            self._choice_graphs = json.load(f)
            self._choice_graphs = [[g[0] for g in graph_set] for graph_set in self._choice_graphs]
        if self._p.get("replace.entities", False):
            self._choice_graphs = [[graph.replace_first_entity(g) for g in graph_set] for graph_set in
                                   self._choice_graphs]
            self._silver_graphs = [[(graph.replace_first_entity(g[0]), g[1],) for g in graph_set] for graph_set in
                                   self._silver_graphs]

        self._choice_graphs = [[graph.normalize_tokens(g) for g in graph_set] for graph_set in
                               self._choice_graphs]
        self._silver_graphs = [[(graph.normalize_tokens(g[0]), g[1],) for g in graph_set] for graph_set in
                               self._silver_graphs]
        assert len(self._dataset_tagged) == len(self._choice_graphs) == len(self._silver_graphs)
        super(WebQuestions, self).__init__(**kwargs)

    def _get_samples(self, questions):
        indices = self._get_sample_indices(questions)
        return self._get_indexed_samples_separate(indices) \
            if self._p.get('each.separate', False) else self._get_indexed_samples(indices)

    def _get_sample_indices(self, questions):
        indices = [q_obj['index'] for q_obj in questions
                   if any(len(g) > 1 and g[1][2] > self._p.get("f1.samples.threshold", 0.5)
                          for g in self._silver_graphs[q_obj['index']]) and self._choice_graphs[q_obj['index']]
                   ]
        return indices

    def _get_indexed_samples(self, indices):
        assert self._p.get("max.silver.samples", 15) < self._p.get("max.negative.samples", 30)
        graph_lists = []
        targets = []
        for index in indices:
            graph_list = self._silver_graphs[index]
            negative_pool = [n_g for n_g in self._choice_graphs[index]
                             if all(n_g.get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]
            if len(graph_list) > self._p.get("max.silver.samples", 15):
                graph_list = [graph_list[i] for i in np.random.choice(range(len(graph_list)),
                                                                      self._p.get("max.silver.samples", 15),
                                                                      replace=False)]
            graph_list, target = self._instance_with_negative(graph_list, negative_pool)
            graph_lists.append(graph_list)
            targets.append(target)
        return graph_lists, np.asarray(targets)

    def _instance_with_negative(self, graph_list, negative_pool):
        negative_pool_size = self._p.get("max.negative.samples", 30) - len(graph_list)
        instance = graph_list[:]
        if negative_pool:
            instance += [(n_g,) for n_g in np.random.choice(negative_pool,
                                                            negative_pool_size,
                                                            replace=len(negative_pool) < negative_pool_size)]
        else:
            instance += [({'edgeSet': []},)] * negative_pool_size
        np.random.shuffle(instance)
        if self._p.get("target.dist", False):
            target = softmax([g[1][2] if len(g) > 1 else 0.0 for g in instance])
        else:
            target = np.argmax([g[1][2] if len(g) > 1 else 0.0 for g in instance])
        instance = [el[0] for el in instance]
        return instance, target

    def _get_indexed_samples_separate(self, indices):
        graph_lists = []
        targets = []
        for index in indices:
            graph_list = self._silver_graphs[index]
            negative_pool = [n_g for n_g in self._choice_graphs[index]
                             if all(n_g.get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]
            for g in graph_list:
                if len(g) > 1 and g[1][2] > self._p.get("f1.samples.threshold", 0.5):
                    instance, target = self._instance_with_negative([g], negative_pool)
                    graph_lists.append(instance)
                    targets.append(target)
        return graph_lists, np.asarray(targets, dtype='int32')

    def get_training_samples(self):
        """
        Get a set of training samples. A tuple is returned where the first element is a list of
        graph sets and the second element is a list of indices. An index points to the correct graph parse
        from the corresponding graph set. Graph sets are all of size 30, negative graphs are subsampled or
        repeatedly sampled if there are more or less negative graphs respectively.
        Graph are stored in triples, where the first element is the graph.

        :return: a set of training samples.
        """
        return self._get_samples(self._questions_train)

    def get_dataset_tokens(self):
        """
        Generate a list of tokens that appear in the training data.

        :return: list of lists of tokens
        """
        return [[w for w, _, _ in q] for q in self._dataset_tagged]

    def get_validation_samples(self):
        """
        See the documentation for get_training_samples

        :return: a set of validation samples distinct from the training samples.
        """
        return self._get_samples(self._questions_val)

    def get_training_generator(self, batch_size):
        """
        Get a set of training samples as a cyclic generator. Negative samples are generated randomly at
        each step.
        Warning: This generator is endless, make sure you have a stopping condition.

        :param batch_size: The size of a batch to return at each step
        :return: a generation that continuously returns batch of training data.
        """
        indices = self._get_sample_indices(self._questions_train)
        for i in itertools.cycle(range(0, len(indices), batch_size)):
            batch_indices = indices[i:i + batch_size]
            yield self._get_indexed_samples(batch_indices)

    def get_validation_with_gold(self):
        """
        Return the validation set with gold answers.
        Returned is a tuple where the first element is a list of graph sets and the second is a list of gold answers.
        Graph sets are of various length and include all possible valid parses of a question, gold answers is a list
        of lists of answers for each qustion. Each answer is a string that might contain multiple tokens.

        :return: a tuple of graphs to choose from and gokd answers
        """
        graph_lists = []
        gold_answers = []
        for q_obj in self._questions_val:
            index = q_obj['index']
            graph_list = self._choice_graphs[index]
            gold_answer = [e.lower() for e in get_answers_from_question(q_obj)]
            graph_lists.append(graph_list)
            gold_answers.append(gold_answer)
        return graph_lists, gold_answers

    def get_train_sample_size(self):
        """
        Compute the size of the training sample with the current settings

        :return: size of the training sample.
        """
        return len(self._get_sample_indices(self._questions_train))

    def get_dataset_size(self):
        """
        Get the size of the complete available dataset.

        :return: size of the dataset.
        """
        return len(self._dataset_tagged)

    def extract_question_entities(self):
        """
        Extracts from each question of the complete dataset a set of entities.

        :return: a list of lists of ordered entities as lists of tokens
        """
        return [graph.extract_entities(tagged_question) for tagged_question in self._dataset_tagged]


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


def softmax(x):
    """
    Compute softmax non-linearity on a vector

    :param x: vector input
    :return: vector output of the same dimension
    """
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
