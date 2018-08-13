import pytest

import torch

from questionanswering.models import modules


def test_batch_similarity():
    questions = torch.randn(8, 16)
    graphs = torch.randn(8, 50, 16)
    predictions = modules.batchmv_cosine_similarity(graphs, questions)
    assert predictions.size() == (8, 50)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
