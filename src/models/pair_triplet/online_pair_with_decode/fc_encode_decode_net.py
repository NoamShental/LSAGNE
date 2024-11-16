from typing import List

from torch import nn as nn

from src.configuration import config
from old_files.fully_connected_embedding_net import EmbeddingNet


class FullyConnectedEncodeDecodeNet(nn.Module):
    def __init__(self, input_size: int, encode_dims: List[int], embedding_dim: int, decode_dims: List[int]):
        super().__init__()
        self.encode_net = EmbeddingNet(input_size=input_size, inner_dims=encode_dims, embedding_dim=embedding_dim)

        inner_layers = [
            nn.Linear(embedding_dim, decode_dims[0]).type(config.torch_numeric_precision_type),
            nn.ReLU()
        ]
        for i in range(1, len(decode_dims)):
            inner_layers.append(nn.Linear(decode_dims[i-1], decode_dims[i]))
            inner_layers.append(nn.ReLU())

        self.decode_net = nn.Sequential(
            *inner_layers,
            nn.Linear(decode_dims[-1], input_size)
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedding = self.encode_net(x)
        decoded = self.decode_net(embedding)
        return x, embedding, decoded

    def get_embedding(self, x):
        return self.encode_net(x)
