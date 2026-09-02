import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_length: int,
        number_of_heads: int = 1,
    ):
        super().__init__()

        if d_model % number_of_heads != 0:
            raise ValueError("d_model must be divisible by number_of_heads")

        self.d_model = d_model
        self.number_of_heads = number_of_heads
        self.head_dim = d_model // number_of_heads

        self.query = nn.Linear(
            d_model,
            d_model,
            bias=False,
            )

        self.key = nn.Linear(
            d_model,
            d_model,
            bias=False,
            )

        self.value = nn.Linear(
            d_model,
            d_model,
            bias=False,
            )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
            )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x, padding_mask: torch.Tensor | None = None,):

        batch_size = x.size(0)
        context_length = x.size(1)

        # -------------------------------------------------
        # Project x into Q, K and V
        # -------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # -------------------------------------------------
        # Split into heads
        # -------------------------------------------------

        Q = Q.view(
            batch_size,
            context_length,
            self.number_of_heads,
            self.head_dim,
        )

        K = K.view(
            batch_size,
            context_length,
            self.number_of_heads,
            self.head_dim,
        )

        V = V.view(
            batch_size,
            context_length,
            self.number_of_heads,
            self.head_dim,
        )

        # -------------------------------------------------
        # Move heads before sequence dimension
        # -------------------------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(
            self.head_dim
        )

        scores = scores.masked_fill(
            self.mask[
                :context_length,
                :context_length,
            ],
            float("-inf"),
        )

        if padding_mask is not None:
            scores = scores.masked_fill(
                ~padding_mask[:, None, :],
                float("-inf"),
            )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        # -------------------------------------------------
        # Weighted values
        # -------------------------------------------------

        output = attention @ V

        # -------------------------------------------------
        # Put heads back together
        # -------------------------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            context_length,
            self.d_model,
        )


        # -------------------------------------------------
        # Output projection
        # -------------------------------------------------

        output = self.output_projection(output)


        return output
