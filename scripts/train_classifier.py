from pathlib import Path

import torch

from src.infra.dataset.classification import TextClassificationDataset
from src.infra.dataset.util import split_tokens, get_batch, build_label, build_text
from src.infra.tokenizer import CharacterTokenizer
from src.infra.dataset.autoregressive import AutoRegressiveDataset

from src.task.config import ModelConfig,TrainingConfig

from src.task.model import TinyTransformerLM, TinyClassifier
from src.task.training.classification import train
import pandas as pd
import numpy as np
from src.task.training.util import TrainingLogger, count_parameters, separator



# Custom Logger
logger = TrainingLogger(
    "logs/text_classifier.log"
    )


# Configs
separator("Configuration")

torch.manual_seed(42)
# Configs
model_config = ModelConfig()
training_config = TrainingConfig()

# Overrides
model_config.context_length = 384
model_config.number_of_heads = 2

training_config.num_steps = 10 
training_config.eval_interval = 1
training_config.num_train_samples = None

logger.log(model_config)
logger.log(training_config)

# Load the corpus
separator("Corpus")

TRAIN_PATH = Path("data/ag-news/train.csv")
TEST_PATH = Path("data/ag-news/test.csv")
checkpoint = "mh_tiny_text_classifier.pt"

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
test_labels = build_label(test_corpus)


# Tokens
separator("Tokenization")

tokenizer = CharacterTokenizer("".join(train_text))

CLASS_TOKENIZER_PATH = Path("checkpoints/ag_news_tokenizer.json")
tokenizer.save(CLASS_TOKENIZER_PATH)

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