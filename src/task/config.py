from dataclasses import dataclass

@dataclass  #thank you v3.7
class ModelConfig:

    context_length: int = 64

    d_model: int = 64       #model dim
    d_ff: int = 256         #feed forward dim
    num_layers: int = 2     #transformer blocks

    dropout: float = 0.1    #percent. avoid overfitting, break emerging biases


@dataclass
class TrainingConfig:

    batch_size: int = 32
    learning_rate: float = 0.001        #the value added/subtracted to gradients
    num_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20