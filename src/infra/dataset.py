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



class TextClassificationDataset:

    def __init__(
        self,
        texts,
        labels,
        tokenizer: CharacterTokenizer,
    ):
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have the same length... why ?"
            )

        self.texts = texts

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        encoded = [
            tokenizer.encode(text)
            for text in texts
        ]

        self.tokens, self.padding_mask = tokenizer.pad(
            encoded
        )

        self.tokens = torch.tensor(
            self.tokens,
            dtype=torch.long,
        )

        self.padding_mask = torch.tensor(
            self.padding_mask,
            dtype=torch.bool,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        return (
            self.tokens[index],
            self.padding_mask[index],
            self.labels[index],
        )


def split_tokens(
    tokens: torch.Tensor,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split a token sequence into train, validation and test sets.

    The sequence is split chronologically rather than randomly.
    This prevents future tokens from leaking into the training set.
    """

    if tokens.ndim != 1:
        raise ValueError(
            "tokens must be a 1-dimensional tensor"
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1"
        )

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1"
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1"
        )

    n = len(tokens)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]

    return (
        train_tokens,
        val_tokens,
        test_tokens,
    )


def get_batch(
    dataset: AutoRegressiveDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from an autoregressive dataset."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )

    indices = torch.randint(
        low=0,
        high=len(dataset),
        size=(batch_size,),
    )

    inputs = []
    targets = []

    for index in indices:

        input_ids, target_ids = dataset[index.item()]

        inputs.append(input_ids)
        targets.append(target_ids)

    return (
        torch.stack(inputs),
        torch.stack(targets),
    )