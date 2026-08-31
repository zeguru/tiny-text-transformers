"""
Attention
    → tokens exchange information

Feed Forward
    → each token independently transforms
      its context-aware representation
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),       #d_ff is the hidden dimension of the feed forward network, usually larger than d_model
            nn.GELU(),                      #Activation function.. Gaussian Error Linear Unit (GELU) is a smooth approximation of the ReLU function, which allows for better gradient flow during training.
            nn.Linear(d_ff, d_model),       #collapses the hidden dimension back to the original model dimension, ensuring that the output has the same shape as the input
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)