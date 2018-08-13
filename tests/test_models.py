import pytest
import json
import random

import numpy as np
import torch
from torch import nn as nn

import fackel

from questionanswering.construction import sentence
from questionanswering import _utils
from questionanswering.models.lexical_baselines import OneEdgeModel, STAGGModel, PooledEdgesModel
from questionanswering.models.modules import ConvWordsEncoder
from questionanswering.models import vectorization as V, losses

wordembeddings, word2idx = V.extend_embeddings_with_special_tokens(
    *_utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.100d.txt")
)
with open(_utils.RESOURCES_FOLDER + "../data/generated/webqsp.examples.train.silvergraphs.02-12.el.unittests.json") as f:
    training_dataset = json.load(f,  object_hook=sentence.sentence_object_hook)


def test_one_edge_model():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = OneEdgeModel(encoder)
    criterion = nn.CrossEntropyLoss()

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion
    )

    train_questions = V.encode_batch_questions(training_dataset, word2idx)[..., 0, :]
    train_edges = V.encode_batch_graphs(training_dataset, word2idx)[..., 0, 0, :]

    container.train(train=(train_questions, train_edges), train_targets=np.zeros(len(training_dataset), dtype=np.int32))


def test_pool_edges_model():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = PooledEdgesModel(encoder)
    criterion = nn.MultiMarginLoss()

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion
    )

    selected_questions = [s for s in training_dataset if any(scores[2] > 0.0 for g, scores in s.graphs)]
    targets = np.zeros((len(selected_questions)), dtype=np.int32)
    for qi, q in enumerate(selected_questions):
        random.shuffle(q.graphs)
        targets[qi] = np.argsort([g.scores[2] for g in q.graphs])[::-1][0]

    train_questions = V.encode_batch_questions(selected_questions, word2idx)[..., 0, :]
    train_edges = V.encode_batch_graphs(selected_questions, word2idx)[..., 0, :]

    container.train(train=(train_questions, train_edges), train_targets=targets)


def test_stagg_model():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = STAGGModel(encoder)
    criterion = nn.CrossEntropyLoss()

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion
    )

    train_questions = V.encode_batch_questions(training_dataset, word2idx)
    train_edges = V.encode_batch_graphs(training_dataset, word2idx)[..., 0, :, :]
    train_features = V.encode_structural_features(training_dataset)

    container.train(train=(train_questions, train_edges, train_features), train_targets=np.zeros(len(training_dataset), dtype=np.int32))


def test_metrics():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = PooledEdgesModel(encoder)
    criterion = nn.MultiMarginLoss()

    def metrics(targets, predictions, validation=False):
        _, predicted_targets = torch.topk(predictions, 1, dim=-1)
        # _, targets = torch.topk(targets, 1, dim=-1)
        predicted_targets = predicted_targets.squeeze(1)
        cur_acc = torch.sum(predicted_targets == targets).float()
        cur_acc /= predicted_targets.size(0)
        cur_f1 = 0.0

        if validation:
            for i, q in enumerate(training_dataset):
                if i < predicted_targets.size(0):
                    idx = predicted_targets.data[i]
                    if idx < len(q.graphs):
                        cur_f1 += q.graphs[idx].scores[2]
            cur_f1 /= targets.size(0)
        return {'acc': cur_acc.data[0], 'f1': cur_f1}

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion,
        metrics=metrics
    )

    selected_questions = [s for s in training_dataset if any(scores[2] > 0.0 for g, scores in s.graphs)]
    targets = np.zeros((len(selected_questions)), dtype=np.int32)
    for qi, q in enumerate(selected_questions):
        random.shuffle(q.graphs)
        targets[qi] = np.argsort([g.scores[2] for g in q.graphs])[::-1][0]

    train_questions = V.encode_batch_questions(selected_questions, word2idx)[..., 0, :]
    train_edges = V.encode_batch_graphs(selected_questions, word2idx)[..., 0, :]

    container.train(train=(train_questions, train_edges), train_targets=targets,
                    dev=(train_questions, train_edges), dev_targets=targets)


if __name__ == '__main__':
    test_metrics()
    pytest.main(['-v', __file__])
