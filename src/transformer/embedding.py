"""
Token embedding layer.

Transforms token IDs into dense, learnable vectors.
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Token Embedding

    Input:
        (batch_size, context_length)

    Output:
        (batch_size, context_length, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)