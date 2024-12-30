from dataclasses import dataclass

import torch
import torch.nn as nn

from Transformer.attention_components import MultiHeadAttention
from Transformer.position_embedding import LearnablePositionalEncoding, PositionalEncoding


@dataclass
class TransformerConfig:
    embed_dim: int
    window_size: int
    vocab_size: int

    attention_head_size: int
    attention_layer_size: int
    hidden_dropout_prob: float

    inference_mode: bool
    device: torch.device

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerBlocks(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embed_dim)

        self.attention = MultiHeadAttention(window_size=config.window_size, embed_dim=config.embed_dim,
                                            num_heads=config.attention_head_size, mask=config.inference_mode)

        self.feed_forward = FeedForward(hidden_size=config.embed_dim, intermediate_size=config.embed_dim * 4,
                                        hidden_dropout_prob=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):

        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = x + shortcut

        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = LearnablePositionalEncoding(config.vocab_size, config.window_size, config.embed_dim, config.device)
        self.encoder = TransformerBlocks(config)

        self.layers = nn.ModuleList([TransformerBlocks(config) for _ in range(config.attention_layer_size)])

        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        encoded_val = self.final_norm(x)

        logits = self.output_layer(encoded_val)

        return logits
