"""
Single-head causal self-attention.

Learns relationships between tokens in the input sequence.

Input Representation (X)
        │
        ▼
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
        │
        ▼
    Scores = Q @ Kᵀ
        │
        ▼
    Scale by √d
        │
        ▼
    Apply Causal Mask
        │
        ▼
    Softmax
        │
        ▼
    Attention Weights
        │
        ▼
Output = Attention Weights @ V

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Single-head causal self-attention.

    Input:
        (batch_size, context_length, d_model)

    Output:
        (batch_size, context_length, d_model)
    """

    def __init__(self, d_model: int, context_length: int,):
        super().__init__()

        # --------------------------------------------------
        # Learnable projection layers.
        #
        # Each nn.Linear layer contains a learnable weight
        # matrix that projects the input representation into
        # a different vector space.
        #
        # Transformer paper notation:
        #
        #   Query Projection -> Wq
        #   Key Projection   -> Wk
        #   Value Projection -> Wv
        #
        # During the forward pass:
        #
        #   Q = X @ Wq
        #   K = X @ Wk
        #   V = X @ Wv
        #
        # In PyTorch, the multiplication is performed by
        # calling the corresponding nn.Linear layer.
        # --------------------------------------------------

        self.query = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.key = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.value = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        # --------------------------------------------------
        # Causal Mask
        #
        # Prevents tokens from attending to future tokens.
        #
        # The mask is created once during initialization
        # and reused during every forward pass.
        # --------------------------------------------------

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



    def forward(self, 
            x: torch.Tensor, 
            padding_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input representation.

            Shape:
                (batch_size, context_length, d_model)

        padding_mask
            Optional mask indicating which positions contain
            real tokens.

            True  -> real token
            False -> padding

            Shape:
                (batch_size, context_length)
        """

        # --------------------------------------------------
        # Step 1
        # Create Query (Q), Key (K) and Value (V)
        #
        # Mathematically:
        #
        #   Q = X @ Wq
        #   K = X @ Wk
        #   V = X @ Wv
        #
        # where X is the input representation and Wq, Wk and
        # Wv are the learnable weight matrices contained in
        # the projection layers above.
        # --------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # --------------------------------------------------
        # Step 2
        # Compute Attention Scores
        #
        # Mathematically:
        #
        #   Scores = Q @ Kᵀ
        #
        # Every Query vector is compared with every Key vector
        # using the dot product, producing one attention score
        # for every Query–Key pair.
        #
        # scores[i, j] answers:
        #
        #   "How much should token i attend to token j?"
        # --------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # --------------------------------------------------
        # Step 3
        # Scale Attention Scores
        #
        # Mathematically:
        #
        #           Q @ Kᵀ
        # Scores = --------
        #            √d
        #
        # where d is the embedding dimension (d_model).
        #
        # Scaling keeps the magnitude of the attention
        # scores relatively stable regardless of the
        # embedding dimension.
        # --------------------------------------------------

        scores = scores / math.sqrt(Q.size(-1))

        # --------------------------------------------------
        # Step 4
        # Apply Causal Mask
        #
        # Prevent each token from attending to future tokens.
        #
        # The masked positions are assigned negative infinity.
        # After applying softmax, they will receive a
        # probability of zero.
        # --------------------------------------------------

        context_length = scores.size(-1)

        # --------------------------------------------------
        # Step 4
        # Apply Causal Mask
        # "Don't look into the future."
        # --------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:context_length, :context_length],
            float("-inf"),
            )

        # --------------------------------------------------
        # Apply Padding Mask
        #
        # True  -> real token
        # False -> PAD
        #
        # We mask the KEY positions, meaning:
        #
        #   "Do not allow any token to attend
        #    to a PAD token."
        # --------------------------------------------------

        if padding_mask is not None:
            scores = scores.masked_fill(
                ~padding_mask[:, None, :],
                float("-inf"),
            )
        # --------------------------------------------------
        # Step 5
        # Apply Softmax
        #
        # Convert the attention scores into probabilities.
        #
        # Softmax is applied independently to each row of
        # the attention score matrix so that every row sums
        # to one.
        #
        # Each probability represents how much attention a
        # Query token pays to each Key token.
        # --------------------------------------------------

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        # --------------------------------------------------
        # Step 6
        # Compute the Weighted Sum
        #
        # Mathematically:
        #
        #   Output = Attention Weights @ V
        #
        # The attention weights determine how much
        # information to retrieve from every Value vector.
        #
        # Each output vector is a weighted combination of
        # all Value vectors visible to the current Query.
        # --------------------------------------------------

        output = attention @ V

        return output
