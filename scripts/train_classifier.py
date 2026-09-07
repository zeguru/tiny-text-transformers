from pathlib import Path

import torch

from src.infra.dataset.classification import TextClassificationDataset
from src.infra.dataset.util import split_tokens, get_batch
from src.infra.tokenizer import CharacterTokenizer
from src.infra.dataset.autoregressive import AutoRegressiveDataset

from src.task.config import ModelConfig,TrainingConfig

from src.task.model import TinyTransformerLM, TinyClassifier
from src.task.training.classification import train
import pandas as pd
import numpy as np



def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

separator("Helpers")

# build text by joining the title and description columns
def build_text(data: pd.DataFrame) -> pd.Series:
    return (
        data["Title"].fillna("").astype(str)
        + " "
        + data["Description"].fillna("").astype(str)
    )

def build_label(data: pd.DataFrame) -> np.ndarray:
    return data["Class Index"].to_numpy() - 1



# Configs
separator("Configuration")

# torch.manual_seed(42)

model_config = ModelConfig()
training_config = TrainingConfig()

print(model_config)
print(training_config)

# Load the corpus
separator("Corpus")

TRAIN_PATH = Path("data/ag-news/train.csv")
TEST_PATH = Path("data/ag-news/test.csv")
checkpoint = "news-classifier.pt"

# AG News uses these class indexes.
CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

print(f"Train file: {TRAIN_PATH}")
print(f"Test file:  {TEST_PATH}")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

separator("Load corpus")

train_corpus = pd.read_csv(TRAIN_PATH)
if training_config.num_train_samples is not None:
    train_corpus = train_corpus.head(training_config.num_train_samples)

test_corpus = pd.read_csv(TEST_PATH)

train_text = build_text(train_corpus)
test_text = build_text(test_corpus)

train_labels = build_label(train_corpus)
print(f"Train labels: {train_labels}")
test_labels = build_label(test_corpus)
print(f"Test labels: {test_labels}")


# Tokens
separator("Tokenization")

tokenizer = CharacterTokenizer("".join(train_text))
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")


# Datasets
separator("Datasets")

train_dataset = TextClassificationDataset(
    texts=train_text,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=model_config.context_length
    )

val_dataset = TextClassificationDataset(
    texts=test_text,
    labels=test_labels,
    tokenizer=tokenizer,
    max_length=model_config.context_length
    )

print(f"Train examples: {len(train_dataset):,}")

# Models
separator("Transformer")

model = TinyClassifier(
    config=model_config,
    vocab_size=tokenizer.vocab_size,
    num_classes=4,
    )

print(model)


# Train........
separator("Training")

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    training_config=training_config,
    )


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