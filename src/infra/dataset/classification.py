
import torch
from src.infra.tokenizer import CharacterTokenizer


class TextClassificationDataset:

    def __init__(
        self,
        texts,
        labels,
        tokenizer: CharacterTokenizer,
        max_length,             #we truncate anything after this length... we got a budget to keep ;-)
    ):
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have the same length"
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
            encoded,
            max_length=max_length,
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



