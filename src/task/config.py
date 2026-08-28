from dataclasses import dataclass

@dataclass
class ModelConfig:

    context_length: int = 64

    d_model: int = 64
    d_ff: int = 256
    num_layers: int = 2

    dropout: float = 0.1


@dataclass
class TrainingConfig:

    batch_size: int = 32
    learning_rate: float = 0.001
    num_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20