from pathlib import Path

import torch

from src.infra.tokenizer import CharacterTokenizer
from src.infra.dataset import AutoRegressiveDataset, split_tokens

from src.task.config import ModelConfig,TrainingConfig

from src.task.model import TinyTransformerLM
from src.task.training import train, evaluate



def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# Runtime configs
corpus = "data/tiny_shakespear.txt"
checkpoint = "tiny-text-transformer.pt"

# Configs
separator("Configuration")

torch.manual_seed(42)

model_config = ModelConfig()
training_config = TrainingConfig()

print(model_config)
print(training_config)


# Load the corpus
separator("Corpus")

data_path = Path(corpus)

text = data_path.read_text(
    encoding="utf-8",
)

print(f"Corpus length: {len(text):,} characters")


# Tokens
separator("Tokenization")

tokenizer = CharacterTokenizer(text)

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Token count: {len(tokens):,}")


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

separator("Data Split")

train_tokens, val_tokens, test_tokens = split_tokens(
    tokens,
    )

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")
print(f"Test tokens:  {len(test_tokens):,}")


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

print(f"Train examples: {len(train_dataset):,}")
print(f"Val examples:   {len(val_dataset):,}")
print(f"Test examples:  {len(test_dataset):,}")


# Models
separator("Model")

model = TinyTransformerLM(
    config=model_config,
    vocab_size=tokenizer.vocab_size,
    )

print(model)


# Train........
separator("Training")

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_config=model_config,
    training_config=training_config,
    )


# Final test evaluation
separator("Test Evaluation")

test_loss = evaluate(
    model=model,
    dataset=test_dataset,
    model_config=model_config,
    training_config=training_config,
)

print(f"Test loss: {test_loss:.4f}")


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