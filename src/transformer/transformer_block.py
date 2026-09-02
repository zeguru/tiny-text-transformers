import torch
import torch.nn as nn

from src.transformer.attention import Attention
from src.transformer.feed_forward import FeedForward
from src.transformer.mutli_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, context_length: int, dropout=0.1,):
        super().__init__()

        self.attention = Attention(d_model, context_length)
        #self.multihead_attention = MultiHeadAttention(d_model, context_length, number_of_heads=4)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask: torch.Tensor | None = None,) -> torch.Tensor:

        attention_output = self.attention(x, padding_mask=padding_mask)
        #attention_output = self.multihead_attention(x, padding_mask=padding_mask)

        x = self.layer_norm_1(
            x + self.dropout(attention_output)
            )

        ffn_output = self.feed_forward(x)

        x = self.layer_norm_2(
            x + self.dropout(ffn_output)        #x is the residual/skip connection... makes sure the original x still influences the learning
            )

        return x