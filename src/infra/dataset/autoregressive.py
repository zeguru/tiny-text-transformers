from __future__ import annotations

import torch

from src.infra.tokenizer import CharacterTokenizer


# Auto Regressive Model: 
# a statistical tool that predicts future values in a sequence by using past values from the same sequence
class AutoRegressiveDataset:
    """Dataset for autoregressive language modeling.

    A token sequence is converted into overlapping
    input/target pairs.

    For a context length of 4:

        tokens:  A B C D E

        input:   A B C D
        target:  B C D E

    Each target sequence is shifted one token
    relative to its corresponding input sequence.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        context_length: int,
    ) -> None:

        if tokens.ndim != 1:
            raise ValueError(
                "tokens must be a 1-dimensional tensor"
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than 0"
            )

        if len(tokens) <= context_length:
            raise ValueError(
                "tokens must contain more elements than "
                "context_length"
            )

        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if index < 0 or index >= len(self):
            raise IndexError(
                "index out of range"
            )

        start = index
        end = start + self.context_length

        input_ids = self.tokens[start:end]      #lag
        target_ids = self.tokens[start + 1:end + 1] 

        return input_ids, target_ids

