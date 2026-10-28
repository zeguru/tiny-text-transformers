from pathlib import Path

import torch
import json

from datetime import datetime
from src.infra.dataset.util import split_tokens, get_batch
from src.infra.tokenizer.character import CharacterTokenizer
from src.infra.tokenizer.bpe import BPETokenizer
from src.infra.dataset.answering import QuestionAnsweringDataset
from torch.utils.data import random_split

from src.task.config import ModelConfig,TrainingConfig

from src.task.model import TinyAnsweringModel
from src.task.training.question_answering import train
from src.task.training.util import TrainingLogger, count_parameters, separator

logger = TrainingLogger(
    "logs/question_answering.log"
    )


checkpoint = "tiny_answering.pt"

# Configs
separator("Configuration")

torch.manual_seed(42)

# Configs
model_config = ModelConfig(
    d_model=128, 
    number_of_heads=2, 
    context_length=1024
    )

training_config = TrainingConfig(
    num_steps=100,
    eval_interval=10,
    batch_size=64
    )

# Overrides
# model_config.context_length=512

# training_config.num_steps = 300 
# training_config.eval_interval = 10

logger.log(model_config)
logger.log(training_config)


# Load the corpus
separator("Corpus")


torch.manual_seed(42)


SQUAD_PATH = Path("data/squad/dev-v1.1.json")

CHAR_TOKENIZER_PATH = Path("checkpoints/qa_tokenizer.json")
BPE_TOKENIZER_PATH = Path("checkpoints/bpe_tokenizer.json")

print(f"Squad file: {SQUAD_PATH}")
print(f"Tokenizer file:  {CHAR_TOKENIZER_PATH}")
print(f"Tokenizer file:  {BPE_TOKENIZER_PATH}")

if not SQUAD_PATH.exists():
    raise FileNotFoundError(f"Squad file not found: {SQUAD_PATH}")

if not CHAR_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Character Tokenizer file not found: {CHAR_TOKENIZER_PATH}")

if not BPE_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"BPE Tokenizer file not found: {BPE_TOKENIZER_PATH}")

char_tokenizer = CharacterTokenizer.load(CHAR_TOKENIZER_PATH)
bpe_tokenizer = BPETokenizer.load(BPE_TOKENIZER_PATH)

#Choose the tokenizer... manual for now
tokenizer = char_tokenizer

vocab_size = tokenizer.vocab_size

with SQUAD_PATH.open("r", encoding="utf-8") as file:
    squad = json.load(file)


# Tokens
records = []

article = next(
    article
    for article in squad["data"]
    if article["title"] == "Kenya"
    )

# for index, article in enumerate(squad["data"]):   #when we need all the data

for paragraph in article["paragraphs"] :
    context = paragraph["context"]
    for qa in paragraph["qas"]:
        for answer in qa["answers"]:
            records.append(
                {   
                    "question": qa["question"],
                    "context": context,
                    "answer": answer["text"],
                    "answer_start": answer["answer_start"],
                }
            )
    

print(f"Number of records: {len(records)}")

print("\n=== Sample record ===")
question = records[0]["question"]
context = records[0]["context"] 
answer_text = records[0]["answer"]
answer_start = records[0]["answer_start"]
answer_end = answer_start + len(answer_text)

print("Context:", context)
print("Question:", question)
print("Answer:", answer_text)
print("Answer Start:", answer_start)
print("Answer End:", answer_end)


separator("Dataset")

dataset = QuestionAnsweringDataset(
    records=records,
    tokenizer=tokenizer,
    max_length=model_config.context_length,         #to manage truncation and padding
    )


n = len(dataset)
train_size = int(0.90 * n)
val_size = int(0.05 * n)
test_size = n - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
    )

logger.log(f"Train examples: {len(train_dataset):,}")
logger.log(f"Val examples:   {len(val_dataset):,}")
logger.log(f"Test examples:  {len(test_dataset):,}")


# Models
separator("Model")

model = TinyAnsweringModel(
    config=model_config,
    vocab_size=tokenizer.vocab_size,
    )

logger.log(model)


# Train........
separator("Training")

print(f"\tStart: {datetime.now()}===")

model = train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    training_config=training_config,
    )

print(f"\tEnd: {datetime.now()}===")

logger.log(f"Parameters: {count_parameters(model):,}")

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