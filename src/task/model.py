"""Single Head transformer"""

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

        print("Initializing TinyTransformerLM")

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
            number_of_heads=config.number_of_heads,
            dropout=config.dropout,
            #padding mask here 
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



class TinyClassifier(nn.Module):

    def __init__(
        self,
        config,
        vocab_size,
        num_classes=4,
    ):
        super().__init__()

        print("Initializing TinyClassifier")

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
            number_of_heads=config.number_of_heads,
            dropout=config.dropout,
        )

        self.head = nn.Linear(
            config.d_model,
            num_classes,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.input_representation(tokens)

        x = self.transformer(
            x,
            padding_mask=padding_mask,
        )

        if padding_mask is None:
            x = x.mean(dim=1)
        else:
            mask = padding_mask.unsqueeze(-1)

            x = x * mask

            token_count = mask.sum(
                dim=1
            ).clamp(min=1)

            x = x.sum(dim=1) / token_count

        logits = self.head(x)

        return logits


class TinyAnsweringModel(nn.Module):

    def __init__(
        self,
        config,
        vocab_size,
        ):
        super().__init__()

        print("Initializing TinyAnsweringModel")

        self.input_representation = InputRepresentation(
            vocab_size=vocab_size,
            context_length=config.context_length, 
            d_model=config.d_model
            )

        self.transformer = Transformer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            context_length=config.context_length,   #max_length used for truncating and padding input
            num_layers=config.num_layers,
            number_of_heads=config.number_of_heads,
            dropout=config.dropout,
            )

        #TWO output projections
        #Start of the answer
        self.start_head = nn.Linear(
            config.d_model, #IN features
            1,              #OUT features
            )

        # Predict where the answer ends.
        self.end_head = nn.Linear(
            config.d_model,
            1,              #why 1 ? should it not be max_length (or max_length)
            )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.input_representation(tokens)

        x = self.transformer(
            x,
            padding_mask=padding_mask,
            )

        start_logits = self.start_head(x).squeeze(-1)
        end_logits = self.end_head(x).squeeze(-1)

        return start_logits, end_logits

    
# Load language modelfrom checkpoint
def load_model(
    checkpoint_path,
    config,
    vocab_size,
    type="GENERATOR"
    ):

    if (type == "GENERATOR"):
        model = TinyTransformerLM(
            config=config,
            vocab_size=vocab_size,
            )
    elif (type == "CLASSIFIER"):
        model = TinyClassifier(
            config=config,
            vocab_size=vocab_size,
            )
    elif (type == "ANSWERING"):
        model = TinyAnsweringModel(
            config=config,
            vocab_size=vocab_size,
            )
    else:
        print("UNKNOWN MODEL")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        )

    model.load_state_dict(checkpoint)
    model.eval()

    return model

