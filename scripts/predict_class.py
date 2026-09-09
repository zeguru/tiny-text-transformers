from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_classifier_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference.classify import TextClassifier
import pandas as pd
import numpy as np
from src.infra.dataset.util import build_text


# Runtime configs
# corpus = "data/tiny_shakespear.txt"
checkpoint = "checkpoints/tiny_text_classifier.pt"

# CLASS_TOKENIZER_PATH = "ag_news_tokenizer.json"
CLASS_TOKENIZER_PATH = Path("checkpoints/ag_news_tokenizer.json")
tokenizer = CharacterTokenizer.load(
    CLASS_TOKENIZER_PATH
    )

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
    ]

# Conf
model_config = ModelConfig()
model_config.context_length = 384


# Load model/checkpoint from file 
model = load_classifier_model(
    checkpoint_path=checkpoint,
    config=model_config,
    vocab_size=tokenizer.vocab_size,
)


# generator
text_classifier = TextClassifier(
    model=model,
    tokenizer=tokenizer,
    context_length=model_config.context_length,
    )


# do generate
prompt_text="World cup 2030 to be in saudi arabia"
output = text_classifier.classify(
    prompt_text
    )

print(prompt_text)
print(CLASS_NAMES[output])