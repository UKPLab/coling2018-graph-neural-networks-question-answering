import pytest
import json

import numpy as np
from torch import nn as nn

import fackel

from questionanswering.construction import sentence
from questionanswering import _utils
from questionanswering.models.modules import ConvWordsEncoder
from questionanswering.models.gnn import GNNModel
from questionanswering.models import vectorization as V


wordembeddings, word2idx = V.extend_embeddings_with_special_tokens(
    *_utils.load_word_embeddings(_utils.RESOURCES_FOLDER + "../../resources/embeddings/glove/glove.6B.100d.txt")
)
with open(_utils.RESOURCES_FOLDER + "../data/generated/webqsp.examples.train.silvergraphs.02-12.el.unittests.json") as f:
    training_dataset = json.load(f,  object_hook=sentence.sentence_object_hook)


def test_encode_structure():
    train_questions = V.encode_batch_graph_structure(training_dataset, word2idx)
    print(train_questions[0])


def test_load_parameters():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = GNNModel(encoder, hp_dropout=0.2)
    criterion = nn.MultiMarginLoss(margin=0.5)

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        save_to_dir="../trainedmodels/",
        early_stopping=5,
        criterion=criterion,
        init_model_weights=True,
        lr_decay=2
    )
    container.save_model()
    container.reload_from_saved()
    assert container._model._gnn._prop_model._dropout.p == 0.2


def test_ggnn():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = GNNModel(encoder)
    criterion = nn.MultiMarginLoss(margin=0.5)

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion,
        init_model_weights=True,
        lr_decay=2
    )

    train_questions = V.encode_batch_questions(training_dataset, word2idx)[..., 0, :]
    train_graphs = V.encode_batch_graph_structure(training_dataset, word2idx)
    targets = np.zeros(len(training_dataset), dtype=np.int32)

    container.train(train=(train_questions, *train_graphs), train_targets=targets)


def test_gnn():
    encoder = ConvWordsEncoder(*wordembeddings.shape)
    encoder.load_word_embeddings_from_numpy(wordembeddings)
    net = GNNModel(encoder, hp_gated=False)
    criterion = nn.MultiMarginLoss(margin=0.5)

    container = fackel.TorchContainer(
        torch_model=net,
        batch_size=8,
        max_epochs=5,
        model_checkpoint=False,
        early_stopping=5,
        criterion=criterion,
        init_model_weights=True,
        lr_decay=2
    )

    train_questions = V.encode_batch_questions(training_dataset, word2idx)[..., 0, :]
    train_graphs = V.encode_batch_graph_structure(training_dataset, word2idx)
    targets = np.zeros(len(training_dataset), dtype=np.int32)

    container.train(train=(train_questions, *train_graphs), train_targets=targets)


if __name__ == '__main__':
    test_gnn()
    # pytest.main(['-v', __file__])
