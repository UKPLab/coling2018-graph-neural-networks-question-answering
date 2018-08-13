import pytest
import json
import random

import numpy as np
import torch
from torch import nn as nn

import fackel

from questionanswering.construction import sentence
from questionanswering import _utils
from questionanswering.models.lexical_baselines import OneEdgeModel
from questionanswering.models.modules import ConvWordsEncoder
from questionanswering.models import vectorization as V
from questionanswering.models import losses


wordembeddings, word2idx = _utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.50d.txt")
with open(_utils.RESOURCES_FOLDER + "../data/generated/webqsp.examples.train.silvergraphs.02-12.el.unittests.json") as f:
    dataset = json.load(f,  object_hook=sentence.sentence_object_hook)


def test_variable_margin_loss():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = OneEdgeModel(encoder)
    criterion = losses.VariableMarginLoss()

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion
    )

    training_dataset = [s for s in dataset if any(scores[2] > 0.0 for g, scores in s.graphs)]
    train_questions = V.encode_batch_questions(training_dataset, word2idx)[..., 0, :]
    train_edges = V.encode_batch_graphs(training_dataset, word2idx)[..., 0, 0, :]
    targets = np.zeros((len(training_dataset), 100))
    for qi, q in enumerate(training_dataset):
        random.shuffle(q.graphs)
        for gi, g in enumerate(q.graphs[:100]):
            targets[qi, gi] = g.scores[2]

    container.train(train=(train_questions, train_edges), train_targets=targets)


if __name__ == '__main__':
    test_variable_margin_loss()
    # pytest.main(['-v', __file__])
