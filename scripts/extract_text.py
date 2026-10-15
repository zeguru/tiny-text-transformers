from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference.extract import TextExtractor
import pandas as pd
import numpy as np
from src.infra.dataset.util import build_text


# Runtime configs
# corpus = "data/tiny_shakespear.txt"
checkpoint = "checkpoints/tiny_answering.pt"

QA_TOKENIZER_PATH = Path("checkpoints/qa_tokenizer.json")
if not QA_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer file not found: {QA_TOKENIZER_PATH}")

tokenizer = CharacterTokenizer.load(QA_TOKENIZER_PATH)


# Conf
model_config = ModelConfig()
# model_config.number_of_heads = 2
model_config.d_model=64
model_config.context_length = 512


model = load_model(
    checkpoint_path=checkpoint,
    config=model_config,
    vocab_size=tokenizer.vocab_size,
    type="ANSWERING"
    )
print("Loaded model \n", model)

# generator
text_extractor = TextExtractor(
    model=model,
    tokenizer=tokenizer,
    context_length=model_config.context_length,
    )

print("Task worker \n", text_extractor)


# do generate
context = "The \"Big Five\" game animals of Africa, that is the lion, leopard, buffalo, rhinoceros, and elephant, can be found in Kenya and in the Masai Mara in particular. A significant population of other wild animals, reptiles and birds can be found in the national parks and game reserves in the country. The annual animal migration occurs between June and September with millions of animals taking part, attracting valuable foreign tourism. Two million wildebeest migrate a distance of 2,900 kilometres (1,802 mi) from the Serengeti in neighbouring Tanzania to the Masai Mara in Kenya, in a constant clockwise fashion, searching for food and water supplies. This Serengeti Migration of the wildebeest is a curious spectacle listed among the 10 Natural Wonders of Africa." 
question = "Where in Kenya can you find the big five"

token_tensor, predicted_start, predicted_end = text_extractor.extract(
    context_text=context,
    question_text=question
    )

print(f"predicted_start : {predicted_start}, predicted_end : {predicted_end}")
if (predicted_start >= predicted_end):
    print("Invalid answer found")

answer_ids = token_tensor[predicted_start:predicted_end + 1]
answer_text = tokenizer.decode(answer_ids.tolist())
print(f"Context: {context}")
print(f"\tQuestion: {question}")
print(f"\tAnswer : {answer_text}")