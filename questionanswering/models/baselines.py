from collections import deque
import nltk
import tqdm
from sklearn import linear_model

from construction import graph
import utils
from . import QAModel, TrainableQAModel
from wikidata import wdaccess
import numpy as np


class LabelOverlapModel(QAModel):

    def __init__(self, **kwargs):
        super(LabelOverlapModel, self).__init__(**kwargs)

    def encode_data_instance(self, instance):
        edge_vectors = deque()
        edge_entities = []
        tokens = instance[0].get("tokens", []) if instance else []
        for g_index, g in enumerate(instance):
            first_edge = graph.get_graph_first_edge(g)
            property_label = wdaccess.property2label.get(first_edge.get('kbID', '')[:-1], utils.unknown_word)
            edge_vectors.append(property_label.split())
            edge_entities.append(int(first_edge['rightkbID'][1:]))
        edge2idx = {e: i for i, e in enumerate(sorted(set(edge_entities), reverse=True))}
        edge_entities = [edge2idx.get(e, 0) for e in edge_entities]
        return tokens, edge_vectors, edge_entities

    def apply_on_instance(self, instance):
        tokens, edge_vectors, edge_entities = self.encode_data_instance(instance)
        predictions = deque()
        for i, edge_vector in enumerate(edge_vectors):
            edge_vector = set(edge_vector)
            score = sum(1 for t in tokens if t in edge_vector) + edge_entities[i]
            predictions.append(score)
        return np.argsort(predictions)[::-1]

    @staticmethod
    def restrict_to_one_entity(data_batch):
        new_data_batch = []
        for instance in data_batch:
            graph_with_entity_id = []
            for g in instance:
                if 'edgeSet' in g:
                    first_edge = g["edgeSet"][0]
                    graph_with_entity_id.append((int(first_edge['rightkbID'][1:]), g))
            first_entity = sorted([i for i, _ in graph_with_entity_id])[0] if graph_with_entity_id else 0
            selected_graphs = [g for i, g in graph_with_entity_id if i == first_entity]
            new_data_batch.append(selected_graphs)
        return new_data_batch


class BagOfWordsModel(LabelOverlapModel, TrainableQAModel):

    def __init__(self, parameters, **kwargs):
        self.question_vocabulary = []
        self.edge_vocabulary = []
        self.threshold = parameters['threshold'] if parameters else 1000
        self._model = linear_model.LogisticRegression()
        self._type2idx = {k: i for i, k in enumerate(wdaccess.sparql_relation.keys())}
        super(BagOfWordsModel, self).__init__(**kwargs)
        self.logger.debug("Parameters: threshold={}".format(self.threshold))

    def encode_data_instance(self, instance):
        tokens, edge_vectors, edge_entities = LabelOverlapModel.encode_data_instance(self, instance)
        tokens = set(tokens)
        tokens_encoded = [int(t in tokens) for t in self.question_vocabulary]
        edges_encoded = deque()
        for g_index, g in enumerate(instance):
            first_edge = graph.get_graph_first_edge(g)
            edge_type = [0]*len(self._type2idx)
            edge_type[self._type2idx.get(first_edge.get('type', 'direct'), 0)] = 1
            edge_vector = [int(t in edge_vectors[g_index]) for t in self.edge_vocabulary]
            edge_vector += edge_type
            edges_encoded.append(edge_vector)
        return tokens_encoded, edges_encoded, edge_entities

    def encode_data_for_training(self, data_with_targets, verbose=False):
        input_set, targets = data_with_targets
        input_encoded, targets_encoded = deque(), deque()
        for index, instance in enumerate(tqdm.tqdm(input_set, ascii=True, disable=(not verbose))):
            tokens_encoded, edges_encoded, edge_entities = self.encode_data_instance(instance)
            for e_index, edge_encoding in enumerate(edges_encoded):
                input_encoded.append(tokens_encoded + edge_encoding + [edge_entities[e_index]])
                targets_encoded.append(1 if targets[index] == e_index else 0)

        return np.asarray(input_encoded, dtype='int8'), np.asarray(targets_encoded, dtype='int8')

    def train(self, data_with_targets, **kwargs):
        encoded_input = [LabelOverlapModel.encode_data_instance(self, instance) for instance in data_with_targets[0]]
        question_fdist = nltk.FreqDist([t for tokens, _, _ in encoded_input for t in tokens])
        edge_fdist = nltk.FreqDist([t for _, edges, _ in encoded_input for edge in edges for t in edge])
        self.question_vocabulary = [t for t, _ in question_fdist.most_common(self.threshold)]
        self.edge_vocabulary = [t for t, _ in edge_fdist.most_common(self.threshold)]

        input_set, targets = self.encode_data_for_training(data_with_targets)
        self._model.fit(input_set, targets)
        self.logger.debug("Model training is finished.")

    def apply_on_instance(self, instance):
        tokens_encoded, edges_encoded, edge_entities = self.encode_data_instance(instance)
        input_encoded = []
        for e_index, edge_encoding in enumerate(edges_encoded):
            input_encoded.append(tokens_encoded + edge_encoding + [edge_entities[e_index]])
        predictions = self._model.predict_proba(input_encoded) if input_encoded else []
        predictions = [p[1] for p in predictions]
        return np.argsort(predictions)[::-1]
