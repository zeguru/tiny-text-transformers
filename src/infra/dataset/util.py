
import torch

from src.infra.dataset.autoregressive import AutoRegressiveDataset
import pandas as pd
import numpy as np


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

# build text by joining the title and description columns
def build_text(data: pd.DataFrame) -> pd.Series:
    return (
        data["Title"].fillna("").astype(str)
        + " "
        + data["Description"].fillna("").astype(str)
    )

def build_label(data: pd.DataFrame) -> np.ndarray:
    return data["Class Index"].to_numpy() - 1
