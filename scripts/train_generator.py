from pathlib import Path

import torch

from src.infra.dataset.util import split_tokens, get_batch
from src.infra.tokenizer import CharacterTokenizer
from src.infra.dataset.autoregressive import AutoRegressiveDataset

from src.task.config import ModelConfig,TrainingConfig

from src.task.model import TinyTransformerLM
from src.task.training.autoregressive import train, evaluate
from src.task.training.util import TrainingLogger, count_parameters, separator


# Custom Logger
logger = TrainingLogger(
    "logs/text_generator.log"
    )


# Runtime configs
corpus = "data/tiny_shakespear.txt"
checkpoint = "mh_tiny_text_generator.pt"

# Configs
separator("Configuration")

torch.manual_seed(42)

# Configs
model_config = ModelConfig(context_length=128, d_model=128, d_ff=512, num_layers=4, number_of_heads=4)

training_config = TrainingConfig()
# Overrides
training_config.num_steps = 10000 
training_config.eval_interval = 1000

logger.log(model_config)
logger.log(training_config)


# Load the corpus
separator("Corpus")

data_path = Path(corpus)

text = data_path.read_text(
    encoding="utf-8",
)

logger.log(f"Corpus length: {len(text):,} characters")


# Tokens
separator("Tokenization")

tokenizer = CharacterTokenizer(text)

TEXT_TOKENIZER_PATH = Path("checkpoints/shakespeare_tokenizer.json")
tokenizer.save(TEXT_TOKENIZER_PATH)

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

logger.log(f"Vocabulary size: {tokenizer.vocab_size}")
logger.log(f"Token count: {len(tokens):,}")


# Train/Val/Test split

separator("Data Split")

train_tokens, val_tokens, test_tokens = split_tokens(
    tokens,
    )

logger.log(f"Train tokens: {len(train_tokens):,}")
logger.log(f"Val tokens:   {len(val_tokens):,}")
logger.log(f"Test tokens:  {len(test_tokens):,}")


# Datasets
separator("Datasets")

train_dataset = AutoRegressiveDataset(
    tokens=train_tokens,
    context_length=model_config.context_length,
    )

val_dataset = AutoRegressiveDataset(
    tokens=val_tokens,
    context_length=model_config.context_length,
    )

test_dataset = AutoRegressiveDataset(
    tokens=test_tokens,
    context_length=model_config.context_length,
    )

logger.log(f"Train examples: {len(train_dataset):,}")
logger.log(f"Val examples:   {len(val_dataset):,}")
logger.log(f"Test examples:  {len(test_dataset):,}")


# Models
separator("Model")

model = TinyTransformerLM(
    config=model_config,
    vocab_size=tokenizer.vocab_size,
    )

logger.log(model)


# Train........
separator("Training")

# train(
#     model=model,
#     train_dataset=train_dataset,
#     val_dataset=val_dataset,
#     training_config=training_config,
#     )

model, steps, train_history, val_history = train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    training_config=training_config,
    )

# Final test evaluation
separator("Test Evaluation")

test_loss = evaluate(
    model=model,
    dataset=test_dataset,
    training_config=training_config,
)

logger.log(f"Test loss: {test_loss:.4f}")



logger.log(f"Parameters: {count_parameters(model):,}")

# --------------------------------------------------
# Save checkpoint
# --------------------------------------------------

separator("Saving")

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = (
    checkpoint_dir / checkpoint
)

torch.save(
    model.state_dict(),
    checkpoint_path,
)

print(f"Checkpoint saved to: {checkpoint_path}")