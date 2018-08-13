import torch
from torch import nn as nn
import numpy as np

from questionanswering.models import modules
from questionanswering.models.modules import batchmv_cosine_similarity


class PropagationModel(nn.Module):

    def __init__(self,
                 hp_emb_size):
        super(PropagationModel, self).__init__()
        self.hp_emb_size = hp_emb_size
        self._update_layer = nn.Sequential(nn.Linear(in_features=hp_emb_size*2, out_features=hp_emb_size),
                                           nn.Tanh()
                                           )

    def forward(self, current_state, edges_m, A_nodes, A_edges):
        nodes_per_graph = current_state.size(1)
        edges_per_node = A_edges.size(-1)

        A_nodes = A_nodes.unsqueeze(-1).expand(-1, -1, -1, current_state.size(-1))
        A_edges = A_edges.unsqueeze(-1).expand(-1, -1, -1, edges_m.size(-1))

        nodes = torch.gather(current_state.unsqueeze(2).expand(-1, -1, edges_per_node, -1),
                             dim=1,
                             index=A_nodes)

        edges = torch.gather(edges_m.unsqueeze(1).expand(-1, nodes_per_graph, -1, -1),
                             dim=2,
                             index=A_edges)
        nodes_mask = (A_nodes != 0).float()
        edges_mask = (A_edges != 0).float()

        new_state = (nodes*nodes_mask + edges*edges_mask).sum(2)
        new_state = self._update_layer(torch.cat((new_state, current_state), dim=-1))
        return new_state


class GatedPropagationModel(nn.Module):

    def __init__(self,
                 hp_emb_size
                 # hp_dropout=0.1
                 ):
        super(GatedPropagationModel, self).__init__()
        self.hp_emb_size = hp_emb_size
        # self.hp_dropout = hp_dropout
        # self._activation_layer = nn.Linear(in_features=hp_emb_size, out_features=hp_emb_size)

        self._update_layer = nn.Sequential(nn.Linear(in_features=hp_emb_size * 2, out_features=hp_emb_size, bias=True),
                                           nn.Sigmoid()
                                           )
        self._reset_layer = nn.Sequential(nn.Linear(in_features=hp_emb_size * 2, out_features=hp_emb_size, bias=True),
                                          nn.Sigmoid()
                                          )

        self._hidden_layer = nn.Sequential(nn.Linear(in_features=hp_emb_size * 2, out_features=hp_emb_size, bias=True),
                                           nn.Tanh()
                                           )
        # self._dropout = nn.Dropout(p=hp_dropout)

    def forward(self, current_state, edges_m, A_nodes, A_edges):
        nodes_per_graph = current_state.size(1)
        edges_per_node = A_edges.size(-1)

        A_nodes = A_nodes.unsqueeze(-1).expand(-1, -1, -1, current_state.size(-1))
        A_edges = A_edges.unsqueeze(-1).expand(-1, -1, -1, edges_m.size(-1))

        nodes = torch.gather(current_state.unsqueeze(2).expand(-1, -1, edges_per_node, -1),
                             dim=1,
                             index=A_nodes)

        edges = torch.gather(edges_m.unsqueeze(1).expand(-1, nodes_per_graph, -1, -1),
                             dim=2,
                             index=A_edges)
        nodes_mask = (A_nodes != 0).float()
        edges_mask = (A_edges != 0).float()

        activation = (nodes*nodes_mask + edges*edges_mask).sum(2)
        activation_current_state = torch.cat((activation, current_state), dim=-1)
        update_gate = self._update_layer(activation_current_state)
        reset_gate = self._reset_layer(activation_current_state)
        new_state = self._hidden_layer(torch.cat((activation, reset_gate * current_state), dim=-1))
        new_state = (1 - update_gate) * current_state + update_gate * new_state

        return new_state


class GNN(nn.Module):

    def __init__(self,
                 hp_in_features,
                 hp_out_features,
                 hp_dropout=0.1,
                 hp_gated=True):
        super(GNN, self).__init__()
        self.hp_in_features = hp_in_features
        self.hp_out_features = hp_out_features
        self.hp_dropout = hp_dropout
        self.hp_gated = hp_gated
        self._steps = 5

        self._prop_model: nn.Module = GatedPropagationModel(hp_out_features)
        self._prop_model: nn.Module = GatedPropagationModel(hp_out_features) if hp_gated else \
            PropagationModel(hp_out_features)

        self._node_layer = nn.Sequential(nn.Linear(in_features=hp_in_features,
                                                   out_features=hp_out_features, bias=True),
                                         nn.Tanh()
                                         )
        self._edge_layer = nn.Sequential(nn.Linear(in_features=hp_in_features,
                                                   out_features=hp_out_features, bias=True),
                                         nn.Tanh()
                                         )
        self.out_edge = nn.Linear(in_features=hp_out_features,
                                  out_features=hp_out_features, bias=False)
        self.in_edge = nn.Linear(in_features=hp_out_features,
                                 out_features=hp_out_features, bias=False)
        self._dropout = nn.Dropout(p=hp_dropout)

    def reset_weights(self):
        self.in_edge.data.normal_(mean=0, std=np.sqrt(1/self.in_edge.size(1)))
        self.out_edge.data.normal_(mean=0, std=np.sqrt(1/self.out_edge.size(1)))
        self.in_edge_bias.data.fill_(0.0)
        self.out_edge_bias.data.fill_(0.0)

    def forward(self, nodes_m, edges_m, A_nodes, A_edges):

        nodes_mask = (A_nodes.sum(-1) != 0).float().unsqueeze(-1).expand(-1, -1, self.hp_out_features)
        edges_mask = (A_edges.sum(-1) != 0).float().unsqueeze(-1).expand(-1, -1, self.hp_out_features)

        current_state = self._node_layer(nodes_m) * nodes_mask
        current_state[:, 1] = 1.0
        edges_m = self._edge_layer(edges_m) * edges_mask
        out_edges = self.out_edge(edges_m)
        in_edges = self.in_edge(edges_m)
        edges_m = torch.cat((out_edges, in_edges), dim=1)
        # edges_m[:, 0] = 0.0
        current_state = self._dropout(current_state)
        edges_m = self._dropout(edges_m)

        for i in range(self._steps):
            current_state = self._prop_model(current_state, edges_m, A_nodes, A_edges)

        graph_vector = current_state[:, 1].contiguous()
        graph_vector = self._dropout(graph_vector)
        return graph_vector
        # return current_state.sum(dim=1)[0]


class GNNModel(nn.Module):

    def __init__(self,
                 tokens_encoder=None,
                 **kwargs
                 ):
        super(GNNModel, self).__init__()
        if tokens_encoder is None:
            tokens_encoder = modules.ConvWordsEncoder(**kwargs)
        self._tokens_encoder: nn.Module = tokens_encoder
        self.output_vector_size = tokens_encoder.output_vector_size // 2
        self._gnn: nn.Module = GNN(self._tokens_encoder._word_embedding.embedding_dim,
                                   tokens_encoder.output_vector_size,
                                   hp_dropout=kwargs.get("hp_dropout", 0.1),
                                   hp_gated=kwargs.get("hp_gated", True)
                                   )
        # self._pool = nn.AdaptiveMaxPool1d(1)

        self._question_layer = nn.Sequential(nn.Linear(in_features=tokens_encoder.output_vector_size,
                                                       out_features=self.output_vector_size),
                                             nn.ReLU()
                                             )
        self._graph_layer = nn.Sequential(nn.Linear(in_features=tokens_encoder.output_vector_size,
                                                    out_features=self.output_vector_size),
                                             nn.ReLU()
                                             )

    def forward(self, questions_m, nodes_m, edges_m, A_nodes, A_edges):

        questions_m = self._tokens_encoder(questions_m)

        graphs_per_sample = edges_m.size(1)
        edges_per_graph = edges_m.size(2)
        predictions_mask = (nodes_m.sum(-1).sum(-1) != 0).float()

        nodes_m = nodes_m.view(-1, nodes_m.size(-1)).long()
        nodes_word_mask = (nodes_m != 0).float().unsqueeze(-1).expand(-1, -1, self._tokens_encoder._word_embedding.embedding_dim)
        nodes_m = self._tokens_encoder._word_embedding(nodes_m)
        nodes_m = nodes_m * nodes_word_mask
        nodes_m = nodes_m.sum(dim=-2)
        # nodes_m = self._pool(nodes_m.transpose(-2, -1)).squeeze(dim=-1)
        # nodes_m = self._tokens_encoder(nodes_m)
        nodes_m = nodes_m.view(-1, edges_per_graph, nodes_m.size(-1))

        edges_m = edges_m.view(-1, edges_m.size(-1)).long()
        edges_word_mask = (edges_m != 0).float().unsqueeze(-1).expand(-1, -1, self._tokens_encoder._word_embedding.embedding_dim)
        edges_m = self._tokens_encoder._word_embedding(edges_m)
        edges_m = edges_m * edges_word_mask
        edges_m = edges_m.sum(dim=-2)
        # edges_m = self._pool(edges_m.transpose(-2, -1)).squeeze(dim=-1)
        # edges_m = self._tokens_encoder(edges_m)
        edges_m = edges_m.view(-1, edges_per_graph, edges_m.size(-1))

        A_nodes, A_edges = A_nodes.view(-1, A_nodes.size(-2), A_nodes.size(-1)).long(), \
                           A_edges.view(-1, A_edges.size(-2), A_edges.size(-1)).long()

        graph_vectors1 = self._gnn(nodes_m, edges_m, A_nodes, A_edges)
        graph_vectors1 = graph_vectors1.view(-1, graphs_per_sample, graph_vectors1.size(-1))

        questions_m = self._question_layer(questions_m)
        graph_vectors1 = self._graph_layer(graph_vectors1)
        predictions = batchmv_cosine_similarity(graph_vectors1, questions_m) * predictions_mask

        return predictions
