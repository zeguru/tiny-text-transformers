from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_classifier_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference.classify import TextClassifier
import pandas as pd
import numpy as np

def build_text(data: pd.DataFrame) -> list[str]:
    return (
        data["Title"].fillna("").astype(str)
        + " "
        + data["Description"].fillna("").astype(str)
    ).tolist()



# Runtime configs
# corpus = "data/tiny_shakespear.txt"
checkpoint = "checkpoints/news-classifier.pt"

TRAIN_PATH = Path("data/ag-news/train.csv")

train_corpus = pd.read_csv(TRAIN_PATH)
train_text = build_text(train_corpus)


tokenizer = CharacterTokenizer("".join(train_text))

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
    ]


# Conf
config = ModelConfig()


# Load model/checkpoint from file 
model = load_classifier_model(
    checkpoint_path=checkpoint,
    config=config,
    vocab_size=tokenizer.vocab_size,
)


# generator
text_classifier = TextClassifier(
    model=model,
    tokenizer=tokenizer,
    context_length=config.context_length,
    )


# do generate
prompt_text="World cup 2030 to be in saudi arabia"
output = text_classifier.classify(
    prompt_text
    )

print(prompt_text)
print(CLASS_NAMES[output])