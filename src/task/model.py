import torch
import torch.nn as nn

from src.transformer.transformer import Transformer
from src.transformer.input_representation import InputRepresentation

class TinyTransformerLM(nn.Module):

    def __init__(
        self,
        config,
        vocab_size,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            config.d_model,
        )

        self.input_representation = InputRepresentation(
            vocab_size=vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            )

        self.transformer = Transformer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            context_length=config.context_length,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.head = nn.Linear(
            config.d_model,
            vocab_size,
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_representation(tokens)

        x = self.transformer(x)

        logits = self.head(x)

        return logits


def load_model(
    checkpoint_path,
    config,
    vocab_size,
):
    model = TinyTransformerLM(
        config=config,
        vocab_size=vocab_size,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model