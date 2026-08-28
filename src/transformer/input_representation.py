"""
Input representation layer.

Combines token embeddings and positional embeddings to produce
the representation that enters the transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_embedding import PositionalEmbedding


class InputRepresentation(nn.Module):
    """
    Input Representation

    Input:
        token_ids
            Shape:
                (batch_size, context_length)

    Output:
        input_representation
            Shape:
                (batch_size, context_length, d_model)

    Formula:
        Input Representation =
            Token Embedding +
            Positional Embedding
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
    ) -> None:
        super().__init__()

        self.token_embedding = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
        )

        self.positional_embedding = PositionalEmbedding(
            context_length=context_length,
            d_model=d_model,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.positional_embedding(token_ids)

        return token_vectors + position_vectors


