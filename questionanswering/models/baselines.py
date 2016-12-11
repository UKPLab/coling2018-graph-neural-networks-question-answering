from collections import deque

import utils
from . import QAModel
from wikidata import wdaccess
import numpy as np
from datasets import evaluation


class LabelOverlapModel(QAModel):

    def __init__(self):
        pass

    @staticmethod
    def encode_data_instance(instance):
        edge_vectors = deque()
        tokens = instance[0].get("tokens", []) if instance else []
        for g_index, g in enumerate(instance):
            first_edge = g["edgeSet"][0] if 'edgeSet' in g else {}
            property_label = wdaccess.property2label.get(first_edge.get('kbID', '')[:-1], utils.unknown_word)
            # property_label += " " + first_edge.get('type', '')
            edge_vectors.append(property_label.split())
        return tokens, edge_vectors

    def test(self, data_with_gold):
        graphs, gold_answers = data_with_gold
        graphs = self.restrict_to_one_entity(graphs)
        predicted_indices = self.apply_on_batch(graphs)
        successes = deque()
        avg_f1 = 0.0
        for i, sorted_indices in enumerate(predicted_indices):
            if sorted_indices:
                retrieved_answers = []
                sorted_indices = deque(sorted_indices)
                while not retrieved_answers and sorted_indices:
                    index = sorted_indices.pop(0)
                    g = graphs[i][index]
                    retrieved_answers = wdaccess.query_graph_denotations(g)
                retrieved_answers = wdaccess.map_query_results(retrieved_answers)
                _, _, f1 = evaluation.retrieval_prec_rec_f1_with_altlabels(gold_answers[i], retrieved_answers)
                if f1:
                    successes.append((i, f1, g))
                avg_f1 += f1
        avg_f1 /= len(gold_answers)
        print("Successful predictions: {} ({})".format(len(successes), len(successes)/len(gold_answers)))
        print("Average f1: {}".format(avg_f1))

    def apply_on_batch(self, data_batch):
        predicted_indices = deque()
        for instance in data_batch:
            predicted_indices.append(self.apply_on_instance(instance))
        return predicted_indices

    def apply_on_instance(self, instance):
        tokens, edge_vectors = self.encode_data_instance(instance)
        predictions = deque()
        for edge_vector in edge_vectors:
            edge_vector = set(edge_vector)
            score = sum(1 for t in tokens if t in edge_vector)
            predictions.append(score)
        return np.argsort(predictions)

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




