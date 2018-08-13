import re
import json

import abc
from typing import List, Dict

import numpy as np
import itertools

from wikidata import scheme

from questionanswering import _utils
from questionanswering.construction import graph
from questionanswering.datasets.dataset import Dataset


class DatasetWithoutNegatives(Dataset, metaclass=abc.ABCMeta):

    def __init__(self, parameters, **kwargs):
        super(DatasetWithoutNegatives, self).__init__(**kwargs)
        self._p = parameters
        self._questions_data = []  # type: List[Dict]
        self._idx2property = []

    def get_training_samples(self, model=None):
        indices = np.random.choice(len(self._questions_data), self._p.get("instances.per.epoch", 10000), replace=False)
        return self._get_indexed_samples(indices)

    def get_question_tokens(self, index):
        tokens = [t.lower() for t in self._questions_data[index]['tokens']]
        if self._p.get("replace.entities", False):
            tokens = graph.replace_entities_in_instance(tokens, [self._questions_data[index]])
        if self._p.get("normalize.tokens", False):
            tokens = [re.sub(r"\d+", "<n>", t.lower()) for t in tokens]
        return tokens

    def _get_indexed_samples(self, indices):
        graph_lists = []
        targets = []
        for index in indices:
            sentence = graph.copy_graph(self._questions_data[index])  # type: Dict
            question_tokens = self.get_question_tokens(index)
            del sentence['tokens']
            graph_list = [(sentence, 1.0 * self._p.get("mult.f1.by", 1.0))]
            negative_pool_size = self._p.get("max.negative.samples", 30) - len(graph_list)
            negative_pool = self._get_negative_instances_for_sentence(negative_pool_size, graph_list)

            negative_pool = [(g, 0.0) for g in negative_pool]
            one_negative = negative_pool[-1]
            instance = graph_list + negative_pool[:-1]
            np.random.shuffle(instance)
            instance = [one_negative] + instance

            target = [g[1] for g in instance] + [0.0] * (self._p.get("max.negative.samples", 30) - len(instance))
            instance = [el[0] for el in instance]

            graph_lists.append((question_tokens, instance))
            targets.append(target)
        return graph_lists, np.asarray(targets)

    def _get_negative_instances_for_sentence(self, pool_size, graph_list):
        sentence = graph_list[0][0]
        negative_pool = []
        for i in range(pool_size):
            neg_graph = graph.copy_graph(sentence)
            for edge in neg_graph['edgeSet']:
                edge['kbID'] = self._idx2property[np.random.randint(len(self._idx2property))] + "v"
                if "label" in edge:
                    del edge["label"]
            negative_pool.append(neg_graph)
        return negative_pool


class SimpleQuestions(DatasetWithoutNegatives):

    def __init__(self, parameters, **kwargs):
        super(SimpleQuestions, self).__init__(parameters, **kwargs)
        self.logger.debug("Loading SimpleQuestions data")
        with open(self._p["path.to.dataset"]) as f:
            self._questions_data = [l.strip().split("\t") for l in f.readlines()]

        self._questions_data = [{"edgeSet": [{"label": l[1].split("/")[-1].replace("_", " "),
                                              "type": "reverse"}],
                                 "tokens": l[3].replace("?", " ?").split()}
                                for l in self._questions_data if len(l) == 4]

        self.logger.debug("Dataset size:{}".format(len(self._questions_data)))
        if self._p.get("instances.per.epoch", 10000) > len(self._questions_data):
            self._p["instances.per.epoch"] = len(self._questions_data)

        self._idx2property = set(scheme.property2label.keys()) - scheme.property_blacklist
        self._idx2property = list(self._idx2property)


class Wikipedia(DatasetWithoutNegatives):
    def __init__(self, parameters, **kwargs):
        super(Wikipedia, self).__init__(parameters, **kwargs)
        self.logger.debug("Loading wikipedia data")
        with open(self._p["path.to.dataset"]) as f:
            self._questions_data = json.load(f, object_hook=dict_to_graph_with_no_vertices)

        for sentence in self._questions_data:
            sentence['edgeSet'] = [e for e in sentence.get("edgeSet", []) if e.get("kbID") != "P0"]
            most_left_entity_id, most_right_entity_id = 0, len(sentence['tokens'])
            if self._p.get("trim.tokens", False):
                entity_ids = [idx for e in sentence['edgeSet'] for p in ["left", "right"] for idx in e[p]]
                if entity_ids:
                    most_left_entity_id = max(min(entity_ids) - self._p.get("trim.context", 1), 0)
                    most_right_entity_id = min(max(entity_ids) + 1 + self._p.get("trim.context", 1), len(sentence['tokens']))
            entities = {}
            for e in sentence['edgeSet']:
                entity_type = "NN" if e.get("kbID") in {"P31", "P106"} else "NNP"
                e['type'] = 'direct'
                e["kbID"] = e.get("kbID") + "v" if e.get("kbID") else None
                for p in ["left", "right"]:
                    entity_tokens = [sentence['tokens'][idx] for idx in e[p]]
                    entities[" ".join(entity_tokens)] = (entity_tokens, entity_type)
                    del e[p]
            sentence['entities'] = list(entities.values())
            if self._p.get("trim.tokens", False):
                sentence['tokens'] = sentence['tokens'][most_left_entity_id:most_right_entity_id]

        self._questions_data = [s for s in self._questions_data if len(s.get('edgeSet', [])) > 0]
        self.logger.debug("Dataset size:{}".format(len(self._questions_data)))
        if self._p.get("instances.per.epoch", 10000) > len(self._questions_data):
            self._p["instances.per.epoch"] = len(self._questions_data)

        self._idx2property = {e.get("kbID") for s in self._questions_data for e in s['edgeSet']}
        self._idx2property = list(self._idx2property)


class WebQuestions(Dataset):
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
        # with open(path_to_dataset["train_tagged"]) as f:
        #     self._dataset_tagged = json.load_word_embeddings(f)
        # self.logger.debug("Tagged: {}".format(len(self._dataset_tagged)))
        # if self._p.get("no.ne.tags", False):
        #     self._dataset_tagged = [[(w, 'O', t) for w, _, t in s] for s in self._dataset_tagged]

        self._questions_train = []
        self._questions_val = []
        self._silver_graphs = []

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

        if len(self._silver_graphs) > 0:
            self.logger.debug("Average number of choices per question: {}".format(np.mean([len(graphs) for graphs in self._silver_graphs])))
            self._silver_graphs = [graph_set if any(len(g) == 3 and type(g[1]) is list and len(g[1]) == 3 and g[1][2] > 0.0 for g in graph_set) else [] for graph_set in self._silver_graphs ]
            self.logger.debug("Real average number of choices per question: {}".format(np.mean([len(graphs) for graphs in self._silver_graphs])))

    def _get_samples(self, questions, model=None):
        indices = self._get_sample_indices(questions)
        return self._get_indexed_samples(indices, model=model)

    def _get_full(self, questions):
        indices = self._get_sample_indices(questions)
        max_silver_samples = self._p.get("max.silver.samples", 15)
        max_negative_samples = self._p.get("max.negative.samples", 30)
        self._p["max.silver.samples"] = 50
        self._p["max.negative.samples"] = max(len(self._silver_graphs[index]) for index in indices)
        full_sample = self._get_indexed_samples(indices)
        self._p["max.silver.samples"] = max_silver_samples
        self._p["max.negative.samples"] = max_negative_samples
        return full_sample

    def _get_sample_indices(self, questions):
        indices = [q_obj['index'] for q_obj in questions
                   if any(len(g) == 3 and len(g[1]) == 3 and g[1][2] > self._p.get("f1.samples.threshold", 0.5)
                          and (not self._p.get("only.with.iclass", False) or any(edge.get("type") == 'iclass' for edge in g[0].get('edgeSet', [])))
                          for g in self._silver_graphs[q_obj['index']])
                   ]
        return indices

    def _get_indexed_samples(self, indices, model=None):
        graph_lists = []
        targets = []
        for index in indices:
            question_tokens = self.get_question_tokens(index)
            graph_list = self._get_question_positive_silver(index)
            negative_pool = self._get_question_negative_silver(index, graph_list)
            negative_pool_size = self._p.get("max.negative.samples", 30) - len(graph_list)
            if model is not None and len(negative_pool) > negative_pool_size:
                negative_pool_scores = model.scores_for_instance((question_tokens, negative_pool))
            else:
                negative_pool_scores = []

            if self._p.get('train.each.separate', False):
                for g in graph_list:
                    instance, target = self._instance_with_negative([g], negative_pool, negative_pool_scores)
                    graph_lists.append((question_tokens, instance))
                    targets.append(target)
            else:
                if len(graph_list) > self._p.get("max.silver.samples", 15):
                    graph_list = [graph_list[i] for i in np.random.choice(range(len(graph_list)),
                                                                          self._p.get("max.silver.samples", 15),
                                                                          replace=False)]
                graph_list, target = self._instance_with_negative(graph_list, negative_pool, negative_pool_scores)
                graph_lists.append((question_tokens, graph_list))
                targets.append(target)
        return graph_lists, targets

    def _get_question_negative_silver(self, index, graph_list=None):
        if graph_list is None:
            graph_list = []
        negative_pool = [n_g[0] for n_g in self._silver_graphs[index]
                         if (len(n_g) < 2 or type(n_g[1]) is not list or len(n_g[1]) < 3 or n_g[1][2] <= self._p.get(
                "f1.samples.threshold", 0.1))
                         and (not self._p.get("only.with.iclass", False) or any(
                edge.get("type") == 'iclass' for edge in n_g[0].get('edgeSet', [])))
                         and all(n_g[0].get('edgeSet', []) != g[0].get('edgeSet', []) for g in graph_list)]
        return negative_pool

    def _get_question_positive_silver(self, index):
        graph_list = [p_g for p_g in self._silver_graphs[index]
                      if len(p_g) == 3 and type(p_g[1]) is list and len(p_g[1]) == 3 and p_g[1][2] > self._p.get("f1.samples.threshold", 0.1)
                      and (not self._p.get("only.with.iclass", False) or any(edge.get("type") == 'iclass' for edge in p_g[0].get('edgeSet', [])))]
        return graph_list

    def _instance_with_negative(self, graph_list, negative_pool, negative_pool_scores):
        assert self._p.get("max.silver.samples", 15) < self._p.get("max.negative.samples", 30) or self._p.get("max.negative.samples", 30) == -1
        negative_pool_size = self._p.get("max.negative.samples", 30) - len(graph_list)
        instance = graph_list[:]
        if len(negative_pool) > 0:
            if len(negative_pool_scores) > 0:
                negative_pool_indices = np.argsort(negative_pool_scores)[::-1][:negative_pool_size]
                negative_pool = [negative_pool[id] for id in negative_pool_indices]
            else:
                if len(negative_pool) > negative_pool_size:
                    negative_pool = np.random.choice(negative_pool,
                                                     negative_pool_size,
                                                     replace=False)
            instance += [(n_g,) for n_g in negative_pool]
        # This is to make sure the zero element is never the target. It is needed for some of the PyTorch losses and shouldn't affect others
        one_negative = instance[-1]
        instance = instance[:-1]
        np.random.shuffle(instance)
        instance = [one_negative] + instance

        target_value_index = 2 if self._p.get("target.value", "f1") else 1 if self._p.get("target.value", "rec") else 0
        target = [g[1][target_value_index] * self._p.get("mult.f1.by", 1.0) if len(g) == 3 else 0.0 for g in instance] \
            + [0.0] * (self._p.get("max.negative.samples", 30) - len(instance))
        instance = [el[0] for el in instance]
        return instance, target

    def get_question_tokens(self, index):
        tokens = [w for w, _, _ in self._dataset_tagged[index]]
        if self._p.get("replace.entities", False) and len(self._silver_graphs) > 0:
            tokens = graph.replace_entities_in_instance(tokens, [g[0] for g in self._get_question_positive_silver(index) if len(g) > 0])
        if self._p.get("normalize.tokens", False):
            tokens = [re.sub(r"\d+", "<n>", t.lower()) for t in tokens]
        return tokens

    def get_training_samples(self, model=None):
        """
        Get a set of training samples. A tuple is returned where the first element is a list of
        graph sets and the second element is a list of indices. An index points to the correct graph parse
        from the corresponding graph set. Graph sets are all of size 30, negative graphs are subsampled or
        repeatedly sampled if there are more or less negative graphs respectively.
        Graph are stored in triples, where the first element is the graph.

        :param model: an optional qamodel that is used to select the negative samples, otherwise random
        :return: a set of training samples.
        """
        return self._get_samples(self._questions_train, model=model)

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
        indices = self._get_sample_indices(self._questions_val)
        each_separate = self._p.get('train.each.separate', False)
        self._p['train.each.separate'] = False
        max_negative_samples = self._p.get("max.negative.samples", 30)
        self._p["max.negative.samples"] = 50
        samples = self._get_indexed_samples(indices)
        self._p["max.negative.samples"] = max_negative_samples
        self._p['train.each.separate'] = each_separate
        return samples

    def get_full_validation(self):
        """
        :return: a set of training samples.
        """
        each_separate = self._p.get('train.each.separate', False)
        self._p['train.each.separate'] = False
        samples = self._get_full(self._questions_val)
        self._p['train.each.separate'] = each_separate
        return samples

    def get_question_tokens_set(self):
        """
        Generate a list of tokens that appear in question in the complete dataset.

        :return: set tokens
        """
        if self._p.get("normalize.tokens", False):
            return {re.sub(r"\d+", "<n>", w.lower()) for q in self._dataset_tagged for w, _, _ in q}
        return {w for q in self._dataset_tagged for w, _, _ in q}

    def get_training_tokens(self):
        """
        Return all question tokens in the training set.

        :return: a list of lists of tokens
        """
        return [self.get_question_tokens(index) for index in self._get_sample_indices(self._questions_train)]

    def get_property_set(self):
        """
        Generate a set of all properties appearing in the dataset.

        :return: set of property ids
        """
        property_set = {e.get("kbID", "")[:-1] for graph_set in self._silver_graphs
                        for g in graph_set for e in g[0].get('edgeSet', []) if 'kbID' in e}
        return property_set

    def get_training_properties_tokens(self):
        """
        Retrieve a list of property tokens that appear in the training data.

        :return: a list of lists of property tokens
        """
        return [scheme.property2label.get(e.get("kbID", "")[:-1], {}).get("label", _utils.unknown_el).split()
                # + " ".join(wdaccess.property2label.get(e.get("kbID", "")[:-1],{}).get("altlabel", [])).split()
                for index in self._get_sample_indices(self._questions_train)
                for g in self._silver_graphs[index] for e in g[0].get('edgeSet', []) if 'kbID' in e]

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


def get_answers_from_question(question_object):
    """
    Retrieve a list of answers from a question as encoded in the WebQuestions dataset.

    :param question_object: A question encoded as a Json object
    :return: A list of answers as strings
    >>> get_answers_from_question({"url": "http://www.freebase.com/view/en/natalie_portman", "targetValue": "(list (description \\"Padm\u00e9 Amidala\\"))", "utterance": "what character did natalie portman play in star wars?"})
    ['Padmé Amidala']
    >>> get_answers_from_question({"targetValue": "(list (description Abduction) (description Eclipse) (description \\"Valentine's Day\\") (description \\"New Moon\\"))"})
    ['Abduction', 'Eclipse', "Valentine's Day", 'New Moon']
    >>> get_answers_from_question({'answers': ['http://www.wikidata.org/entity/Q16759', \
    'http://www.wikidata.org/entity/Q190972','http://www.wikidata.org/entity/Q231093',],'utterance': 'Which actors play in Big Bang Theory?'})
    ['Q16759', 'Q190972', 'Q231093']
    """
    if 'answers' in question_object or "answer" in question_object:
        answers = []
        for a in question_object.get('answers', question_object.get('answer')):
            if a and a.startswith(scheme.WIKIDATA_ENTITY_PREFIX):
                a = a.replace(scheme.WIKIDATA_ENTITY_PREFIX, "")
            answers.append(a)
        return answers
    answers = re.findall("\(description \"?(.*?)\"?\)", question_object.get('targetValue'))
    if len(answers) == 1 and answers[0].startswith("http://"):
        return []
    return answers


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


def dict_to_graph_with_no_vertices(d):
    if 'vertexSet' in d:
        del d['vertexSet']
    return d


def softmax(x):
    """
    Compute softmax non-linearity on a vector

    :param x: vector input
    :return: vector output of the same dimension
    """
    return np.exp(x) / np.sum(np.exp(x))


def f1_to_dist(x):
    """
    Transforms a list of f1 to a probability distribution. 
    
    :param x: a list of f1 scores
    :return: a probability distribution
    >>> f1_to_dist([0.28571,0.0,0.0,0.0])
    array([ 1.,  0.,  0.,  0.])
    >>> f1_to_dist([0.28571,0.18,0.0,0.0])
    array([ 0.61349338,  0.38650662,  0.        ,  0.        ])
    """
    denominator = np.sum(x)
    return np.asarray(x) / denominator


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
