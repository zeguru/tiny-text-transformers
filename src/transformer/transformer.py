import torch
import torch.nn as nn

from src.transformer.transformer_block import TransformerBlock


class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        context_length: int,
        num_layers: int,
        dropout = 0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                context_length=context_length,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, 
                x: torch.Tensor, 
                padding_mask: torch.Tensor | None = None,) -> torch.Tensor:

        for block in self.blocks:
            x = block(x,padding_mask=padding_mask,)

        return x