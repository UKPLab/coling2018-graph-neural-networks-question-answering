import torch
from torch import nn as nn

from questionanswering.models import modules
from questionanswering.models.modules import batchmv_cosine_similarity


class OneEdgeModel(nn.Module):

    def __init__(self,
                 tokens_encoder=None,
                 **kwargs
                 ):
        super(OneEdgeModel, self).__init__()
        if tokens_encoder is None:
            tokens_encoder = modules.ConvWordsEncoder(**kwargs)
        self._tokens_encoder: nn.Module = tokens_encoder

    def forward(self, questions_m, graphs_m):

        question_vector1 = self._tokens_encoder(questions_m)

        edge_vectors1 = self._tokens_encoder(graphs_m.view(-1, graphs_m.size(-1)))
        edge_vectors1 = edge_vectors1.view(-1, graphs_m.size(1), edge_vectors1.size(-1))

        # Batch cosine similarity
        predictions = batchmv_cosine_similarity(edge_vectors1, question_vector1)

        return predictions


class STAGGModel(nn.Module):

    def __init__(self,
                 tokens_encoder=None,
                 **kwargs
                 ):
        super(STAGGModel, self).__init__()
        if tokens_encoder is None:
            tokens_encoder = modules.ConvWordsEncoder(**kwargs)
        self._tokens_encoder: nn.Module = tokens_encoder
        self._weights_layer = nn.Sequential(nn.Linear(in_features=9,
                                                       out_features=1),
                                            nn.ReLU()
                                            )

    def forward(self, questions_m, graphs_m, graphs_features_m):
        graphs_features_m = graphs_features_m.float()

        question_vector1 = self._tokens_encoder(questions_m[..., 0, :])
        question_vector2 = self._tokens_encoder(questions_m[..., 1, :])

        edge_vectors1 = self._tokens_encoder(graphs_m[..., 0, :]
                                             .contiguous()
                                             .view(-1, graphs_m.size(-1)))
        edge_vectors2 = self._tokens_encoder(graphs_m[..., 1, :]
                                             .contiguous()
                                             .view(-1, graphs_m.size(-1)))
        edge_vectors1 = edge_vectors1.view(-1, graphs_m.size(1), edge_vectors1.size(-1))
        edge_vectors2 = edge_vectors2.view(-1, graphs_m.size(1), edge_vectors2.size(-1))

        # Batch cosine similarity
        predictions1 = batchmv_cosine_similarity(edge_vectors1, question_vector1)
        predictions2 = batchmv_cosine_similarity(edge_vectors2, question_vector2)

        graphs_features_m = torch.cat((predictions1.unsqueeze(2), predictions2.unsqueeze(2), graphs_features_m), dim=-1)
        predictions = self._weights_layer(graphs_features_m).squeeze(-1)

        return predictions


class PooledEdgesModel(nn.Module):

    def __init__(self,
                 tokens_encoder=None,
                 **kwargs
                 ):
        super(PooledEdgesModel, self).__init__()
        if tokens_encoder is None:
            tokens_encoder = modules.ConvWordsEncoder(**kwargs)
        self._tokens_encoder: nn.Module = tokens_encoder
        self._pool = self._tokens_encoder._pool

    def forward(self, questions_m, graphs_m, *args):
        question_vector = self._tokens_encoder(questions_m)
        edge_vectors = graphs_m.view(-1, graphs_m.size(-1))

        edge_vectors = self._tokens_encoder(edge_vectors)
        edge_vectors = edge_vectors.view(-1, graphs_m.size(-2), edge_vectors.size(-1))\
            .transpose(-1, -2).contiguous()
        edge_vectors = self._pool(edge_vectors).squeeze(-1)
        edge_vectors = edge_vectors.view(-1, graphs_m.size(1), edge_vectors.size(-1))

        # Batch cosine similarity
        predictions = batchmv_cosine_similarity(edge_vectors, question_vector)

        return predictions



