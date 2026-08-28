"""
Positional embedding layer.

Converts token positions into learnable embedding vectors.
"""

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Input:
        token_ids: (batch_size, context_length)

    Output:
        (batch_size, context_length, d_model)
    """

    def __init__(self, context_length: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=d_model
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        batch_size, context_length = token_ids.shape

        positions = torch.arange(
            context_length,
            device=token_ids.device
        )

        positions = positions.unsqueeze(0).expand(batch_size, context_length)

        return self.embedding(positions)