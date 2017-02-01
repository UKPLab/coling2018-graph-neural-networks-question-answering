import re
import json
import numpy as np
import itertools

from utils import Loggable
from construction import graph
from wikidata import wdaccess


class WebQuestions(Loggable):
    def __init__(self, parameters, **kwargs):
        """
        An object class to access the webquestion dataset. The path to the dataset should point to a folder that
        contains a preprocessed dataset.

        :param path_to_dataset: path to the data set location
        """
        super(WebQuestions, self).__init__(**kwargs)
        self._p = parameters
        path_to_dataset = self._p["path.to.dataset"]
        self.logger.debug("Loading data")
        # Load the tagged version. This part should be always present.
        with open(path_to_dataset["train_tagged"]) as f:
            self._dataset_tagged = json.load(f)
        self.logger.debug("Tagged: {}".format(len(self._dataset_tagged)))

        self._questions_train = []
        self._questions_val = []
        self._silver_graphs = []
        self._choice_graphs = []

        # Load the train questions
        if "train_train" in path_to_dataset:
            with open(path_to_dataset["train_train"]) as f:
                self._questions_train = json.load(f)
            self.logger.debug("Train: {}".format(len(self._questions_train)))
        # Load the validation questions
        if "train_validation" in path_to_dataset:
            with open(path_to_dataset["train_validation"]) as f:
                self._questions_val = json.load(f)
            self.logger.debug("Val: {}".format(len(self._questions_val)))
        # Load the generated graphs
        if "train_silvergraphs" in path_to_dataset:
            with open(path_to_dataset["train_silvergraphs"]) as f:
                self._silver_graphs = json.load(f)
            self.logger.debug("Silver: {}".format(len(self._silver_graphs)))
        # Load the choice graphs. Choice graphs are all graph derivable from each sentence.
        if "train_choicegraphs" in path_to_dataset:
            with open(path_to_dataset["train_choicegraphs"]) as f:
                self._choice_graphs = json.load(f)
            self.logger.debug("Choice: {}".format(len(self._choice_graphs)))

        if len(self._choice_graphs) > 0:
            assert len(self._dataset_tagged) == len(self._choice_graphs)
            self._choice_graphs = [[g[0] for g in graph_set] for graph_set in self._choice_graphs]
            self.logger.debug("Average number of choices per question: {}".format(
                np.mean([len(graphs) for graphs in self._choice_graphs])))
            self.logger.debug("Removing graphs that use disallowed extensions")
            self._choice_graphs = [[g for g in graph_set if graph.if_graph_adheres(g, allowed_extensions=self._p.get("extensions", set()))] for graph_set in self._choice_graphs]
            self.logger.debug("Average number of choices per question: {}".format(
                np.mean([len(graphs) for graphs in self._choice_graphs])))
            self.logger.debug("Adding tokens to graphs")
            for i, graph_set in enumerate(self._choice_graphs):
                tokens = [w for w,_,_ in self._dataset_tagged[i]]
                for g in graph_set:
                    g['tokens'] = tokens

        if len(self._silver_graphs) > 0:
            assert len(self._dataset_tagged) == len(self._silver_graphs)
            self._silver_graphs = [[g for g in graph_set if graph.if_graph_adheres(g[0], allowed_extensions=self._p.get("extensions", set()))] for graph_set in self._silver_graphs]
            self.logger.debug("Average number of choices per question: {}".format(np.mean([len(graphs) for graphs in self._silver_graphs])))
            if self._p.get("use.whitelist", False):
                self.logger.debug("Using only whitelisted relations for training")
                self._silver_graphs = [[g for g in graph_set if all(e.get('type') in {'time', 'v-structure'} or e.get("kbID")[:-1] in wdaccess.property_whitelist for e in g[0].get('edgeSet', []))]
                                       for graph_set in self._silver_graphs]
                self.logger.debug("Average number of choices per question: {}".format(np.mean([len(graphs) for graphs in self._silver_graphs])))
            if self._p.get("max.entity.options", 3) == 1:
                self.logger.debug("Restricting to one entity choice")
                target_entities = [{e.get('rightkbID') for e in instance[np.argmax([g[1][2] if len(g) > 1 else 0.0 for g in instance])][0].get('edgeSet', [])} for instance in self._silver_graphs]
                self._silver_graphs = [[g for g in graph_set if all(e.get('rightkbID') in target_entities[i] for e in g[0].get('edgeSet', []))]
                                       for i, graph_set in enumerate(self._silver_graphs)]
                self.logger.debug("Average number of choices per question: {}".format(np.mean([len(graphs) for graphs in self._silver_graphs])))

        if self._p.get("replace.entities", False):
            self.logger.debug("Replacing entities in questions")
            self._choice_graphs = [[graph.replace_entities(g) for g in graph_set] for graph_set in
                                   self._choice_graphs]
            for graph_set in self._silver_graphs:
                for g in graph_set:
                    g[0] = graph.replace_entities(g[0])
        if self._p.get("normalize.tokens", False):
            self.logger.debug("Normalizing tokens in questions")
            self._choice_graphs = [[graph.normalize_tokens(g) for g in graph_set] for graph_set in
                                   self._choice_graphs]
            for graph_set in self._silver_graphs:
                for g in graph_set:
                    g[0] = graph.normalize_tokens(g[0])

        self.logger.debug("Constructing string representations for entities")
        self._choice_graphs = [[graph.add_string_representations_to_edges(g, wdaccess.property2label, self._p.get("replace.entities", False)) for g in graph_set]
                               for graph_set in self._choice_graphs]
        for graph_set in self._silver_graphs:
            for g in graph_set:
                g[0] = graph.add_string_representations_to_edges(g[0], wdaccess.property2label, self._p.get("replace.entities", False))

    def _get_samples(self, questions):
        indices = self._get_sample_indices(questions)
        return self._get_indexed_samples_separate(indices) \
            if self._p.get('each.separate', False) else self._get_indexed_samples(indices)

    def _get_full(self, questions):
        indices = self._get_sample_indices(questions)
        max_silver_samples = self._p.get("max.silver.samples", 15)
        max_negative_samples = self._p.get("max.negative.samples", 30)
        self._p["max.silver.samples"] = 50
        self._p["max.negative.samples"] = 500
        full_sample = self._get_indexed_samples_separate(indices) \
            if self._p.get('each.separate', False) else self._get_indexed_samples(indices)
        self._p["max.silver.samples"] = max_silver_samples
        self._p["max.negative.samples"] = max_negative_samples
        return full_sample

    def _get_sample_indices(self, questions):
        indices = [q_obj['index'] for q_obj in questions
                   if any(len(g) > 1 and len(g[1]) == 3 and g[1][2] > self._p.get("f1.samples.threshold", 0.5)
                          for g in self._silver_graphs[q_obj['index']]) and (len(self._choice_graphs) == 0 or self._choice_graphs[q_obj['index']])
                   ]
        return indices

    def _get_indexed_samples(self, indices):
        assert self._p.get("max.silver.samples", 15) < self._p.get("max.negative.samples", 30)
        graph_lists = []
        targets = []
        for index in indices:
            graph_list = [p_g for p_g in self._silver_graphs[index]
                          if len(p_g) > 1 and len(p_g[1]) == 3 and p_g[1][2] > self._p.get("f1.samples.threshold", 0.1)]
            if len(self._choice_graphs) > 0:
                negative_pool = [n_g for n_g in self._choice_graphs[index]
                                 if all(n_g.get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]
            else:
                negative_pool = [n_g[0] for n_g in self._silver_graphs[index]
                                 if (len(n_g) < 2 or len(n_g[1]) < 3 or n_g[1][2] <= self._p.get("f1.samples.threshold", 0.1))
                                 and all(n_g[0].get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]

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

    def get_full_training(self):
        """
        :return: a set of training samples.
        """

        return self._get_full(self._questions_train)

    def get_validation_samples(self):
        """
        See the documentation for get_training_samples

        :return: a set of validation samples distinct from the training samples.
        """
        return self._get_samples(self._questions_val)

    def get_full_validation(self):
        """
        :return: a set of training samples.
        """

        return self._get_full(self._questions_val)

    def get_question_tokens(self):
        """
        Generate a list of tokens that appear in question in the complete dataset.

        :return: list of lists of tokens
        """
        return [[w for w, _, _ in q] for q in self._dataset_tagged]

    def get_training_tokens(self):
        """
        Generate a list of tokens that appear in the training data in the questions and in the edge labels.

        :return: list of lists of tokens
        """
        if len(self._choice_graphs) > 0:
            return [[w for w, _, _ in self._dataset_tagged[i]] +
                [w for g in self._choice_graphs[i] for e in g.get('edgeSet', []) for w in e.get('label', '').split()]
                for i in self._get_sample_indices(self._questions_train)]
        return [[w for w, _, _ in self._dataset_tagged[i]] +
                [w for g in self._silver_graphs[i] for e in g[0].get('edgeSet', []) for w in e.get('label', '').split()]
                for i in self._get_sample_indices(self._questions_train)]

    def get_property_set(self):
        """
        Generate a set of all properties appearing in the dataset.

        :return: set of property ids
        """
        property_set = {e.get("kbID", "")[:-1] for graph_set in self._silver_graphs
                for g in graph_set for e in g[0].get('edgeSet', []) if 'kbID' in e}
        if len(self._choice_graphs) > 0:
            property_set = property_set | {e.get("kbID", "")[:-1] for graph_set in self._choice_graphs
             for g in graph_set for e in g[0].get('edgeSet', []) if 'kbID' in e}
        return property_set

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
    if 'answers' in question_object:
        return question_object['answers']
    return re.findall("\(description \"?(.*?)\"?\)", question_object.get('targetValue'))


def get_main_entity_from_question(question_object):
    """
    Retrieve the main Freebase entity linked in the url field

    :param question_object: A question encoded as a Json object
    :return: A list of answers as strings
    >>> get_main_entity_from_question({"url": "http://www.freebase.com/view/en/natalie_portman", "targetValue": "(list (description \\"Padm\u00e9 Amidala\\"))", "utterance": "what character did natalie portman play in star wars?"})
    (['Natalie', 'Portman'], 'URL')
    >>> get_main_entity_from_question({"url": "http://www.freebase.com/view/en/j_j_thomson"})
    (['J', 'J', 'Thomson'], 'URL')
    >>> get_main_entity_from_question({"targetValue": "(list (description Abduction) (description Eclipse) (description \\"Valentine's Day\\") (description \\"New Moon\\"))"})
    ()
    >>> get_main_entity_from_question({"url": "http://www.freebase.com/view/en/j_j_thomson"})
    (['J', 'J', 'Thomson'], 'URL')
    """
    url = question_object.get('url')
    if url:
        if "http://www.freebase.com/view/en/" not in url:
            return [w.title() for w in url.split()], 'URL'
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
