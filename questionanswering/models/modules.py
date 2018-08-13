import numpy as np
import torch
from torch import nn as nn

from questionanswering.models import pooling as P


class ConvWordsEncoder(nn.Module):

    def __init__(self,
                 hp_vocab_size=10,
                 hp_word_emb_size=50,
                 hp_dropout=0.1,
                 hp_conv_size=32,
                 hp_conv_width=3,
                 hp_dilated_conv_depth=1,
                 hp_pooling='max',
                 hp_repeat_cnn=1,
                 hp_add_top_dense_layer=True,
                 **kwargs
                 ):
        super(ConvWordsEncoder, self).__init__()
        self.hp_vocab_size = hp_vocab_size
        self.hp_word_emb_size = hp_word_emb_size
        self.hp_dropout = hp_dropout
        self.hp_conv_size = hp_conv_size
        self.hp_conv_width = hp_conv_width
        self.hp_dilated_conv_depth = hp_dilated_conv_depth
        self.hp_pooling = hp_pooling
        self.hp_repeat_cnn = hp_repeat_cnn
        self.hp_add_top_dense_layer = hp_add_top_dense_layer

        self.output_vector_size = hp_conv_size // 2 if self.hp_add_top_dense_layer else hp_conv_size

        self._dropout = nn.Dropout(p=hp_dropout)

        self._word_embedding = nn.Embedding(hp_vocab_size, hp_word_emb_size, padding_idx=0)
        self._word_embedding.weight.requires_grad = False

        self._nonlinearity = nn.ReLU()

        self._block_conv_in = nn.Sequential(nn.Conv1d(in_channels=hp_word_emb_size,
                                                      out_channels=hp_conv_size,
                                                      kernel_size=hp_conv_width,
                                                      padding=hp_conv_width // 2,
                                                      dilation=1),
                                            self._nonlinearity
                                            )

        self._dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=hp_conv_size,
                          out_channels=hp_conv_size,
                          kernel_size=hp_conv_width,
                          padding=hp_conv_width // 2 * 2 ** (j + 1),
                          dilation=2**(j + 1),
                          bias=True),
                self._nonlinearity
            )
            for j in range(hp_dilated_conv_depth)
        ])

        if self.hp_repeat_cnn > 0:
            self._block_conv_out = nn.Sequential(nn.Conv1d(in_channels=hp_conv_size,
                                                           out_channels=hp_conv_size,
                                                           kernel_size=hp_conv_width,
                                                           padding=hp_conv_width // 2,
                                                           dilation=1),
                                                 self._nonlinearity
                                                 )

        self._pool = {
            'max': nn.AdaptiveMaxPool1d(1),
            'avg': nn.AdaptiveAvgPool1d(1),
            'logsumexp': P.LogSumExpPooling1d()
        }.get(hp_pooling,
              nn.AdaptiveMaxPool1d(1))

        if self.hp_add_top_dense_layer:
            self._semantic_layer = nn.Sequential(nn.Linear(in_features=hp_conv_size,
                                                       out_features=self.output_vector_size),
                                                 self._nonlinearity
                                                 )

    def load_word_embeddings_from_numpy(self, word_embeddings: np.ndarray):
        word_embeddings = torch.from_numpy(word_embeddings).float()
        self._word_embedding.weight = nn.Parameter(word_embeddings)
        self._word_embedding.weight.requires_grad = False

    def forward(self, words_m):
        words_m = words_m.long()
        words_mask = (words_m != 0).float().unsqueeze(-1).expand(-1, -1, self.hp_conv_size).transpose(-2, -1)
        words_m = self._word_embedding(words_m)
        words_m = words_m.transpose(-2, -1).contiguous()

        words_m = self._block_conv_in(words_m)
        words_m = self._dropout(words_m)

        for _ in range(self.hp_repeat_cnn):
            for convlayer in self._dilated_convs:
                words_m = convlayer(words_m)
            words_m = self._block_conv_out(words_m)

        words_m = words_m * words_mask
        words_v = self._pool(words_m).squeeze(dim=-1)
        words_v = self._dropout(words_v)
        if self.hp_add_top_dense_layer:
            words_v = self._semantic_layer(words_v)
        return words_v


def batchmv_cosine_similarity(m, v):
    """
    Computes a cosine similarity between batches of matrices and vectors.

    :param m: a #D tensor that contains a batch of matrices
    :param v: a 2D tensor that contains a batch of vectors
    :return: a 2D tensor with similarity values per vector * matrix row
    """
    predictions = torch.bmm(m, v.unsqueeze(2)).squeeze()
    w1 = torch.norm(m, 2, dim=-1)
    w2 = torch.norm(v, 2, dim=-1, keepdim=True)
    predictions = (predictions / (w1 * w2.expand_as(w1)).clamp(min=10e-8))
    return predictions