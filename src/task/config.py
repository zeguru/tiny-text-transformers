from dataclasses import dataclass

@dataclass  #thank you v3.7
class ModelConfig:

    context_length: int = 64

    d_model: int = 64       #model dim
    d_ff: int = 256         #feed forward dim
    num_layers: int = 2     #transformer blocks
    number_of_heads: int = 1
    dropout: float = 0.1    #percent. avoid overfitting, break emerging biases


@dataclass
class TrainingConfig:

    batch_size: int = 32
    learning_rate: float = 0.001        #the value added/subtracted to gradients
    num_steps: int = 100
    eval_interval: int = 10
    eval_batches: int = 20

    num_train_samples: int | None = None  #None for all, or limit to N samples for testing


#need to connect to config file or cli args
@dataclass
class RuntimeConfig:
    number_of_heads: int = 2
    