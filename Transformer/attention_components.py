from math import sqrt

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, window_size: int, embed_dim: int, num_heads: int, mask: bool):
        super().__init__()
        self.has_mask = mask
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        attention_mask = torch.tril(torch.ones(window_size, window_size)).unsqueeze(0)
        self.register_buffer("mask", attention_mask)

        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_state):
        B, N, E = hidden_state.shape

        queries = self.w_q(hidden_state)
        values = self.w_v(hidden_state)
        keys = self.w_k(hidden_state)

        keys = keys.view(B, N, self.num_heads, self.head_dim)
        values = values.view(B, N, self.num_heads, self.head_dim)
        queries = queries.view(B, N, self.num_heads, self.head_dim)

        # Transpose to B, num_heads, N, head_dim
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        if self.has_mask:
            attn_scores = attn_scores.masked_fill(self.mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores / sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Transpose back to B, N, num_heads, head_dim
        # Concat head_dim to embed_dim
        x = (attn_weights @ values).transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        x = self.output_linear(x)

        return x

