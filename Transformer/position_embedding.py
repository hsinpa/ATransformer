import math

import torch
from torch import nn

def positional_encoding(batch_size: int, seq_len: int, embedding_dim: int):
    position_index = torch.arange(seq_len).unsqueeze(1)
    # Shape (seq_len, 1)

    # this part is static, so calculate once here
    # half size of embedding_dim
    div_term = torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
    div_term = torch.exp(div_term)

    pe = torch.zeros(batch_size, seq_len, embedding_dim)

    # {target_index}::{skip} => For our case, skip 2 index in a loop
    pe[:, :, 0::2] = torch.sin(position_index * div_term)
    pe[:, :, 1::2] = torch.cos(position_index * div_term)

    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, batch_size: int, sequence_size: int, embedding_dim: int):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=0.2)

        pe = positional_encoding(batch_size, sequence_size, embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, input_token):
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_token)
        position_embeddings = self.pe

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, sequence_size: int, embedding_dim: int, device: torch.device):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=0.2)
        self.position_ids = torch.arange(sequence_size, dtype=torch.long).unsqueeze(0).to(device)
        self.position_embeddings = nn.Embedding(sequence_size, embedding_dim)

    def forward(self, input_token):
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_token)
        position_embeddings = self.position_embeddings(self.position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings