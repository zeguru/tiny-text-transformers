from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_model
from src.infra.tokenizer.character import CharacterTokenizer
from src.infra.tokenizer.bpe import BPETokenizer
from src.task.inference.extract import TextExtractor
import pandas as pd
import numpy as np
from src.infra.dataset.util import build_text


# Runtime configs
# corpus = "data/tiny_shakespear.txt"
checkpoint = "checkpoints/tiny_answering.pt"

CHAR_TOKENIZER_PATH = Path("checkpoints/qa_tokenizer.json")
BPE_TOKENIZER_PATH = Path("checkpoints/bpe_tokenizer.json")

if not CHAR_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Char Tokenizer file not found: {CHAR_TOKENIZER_PATH}")

if not BPE_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"BPE Tokenizer file not found: {BPE_TOKENIZER_PATH}")


char_tokenizer = CharacterTokenizer.load(CHAR_TOKENIZER_PATH)
bpe_tokenizer = BPETokenizer.load(BPE_TOKENIZER_PATH)

#Choose the tokenizer... manual for now
tokenizer = char_tokenizer

# Conf
# model_config = ModelConfig()
# # model_config.number_of_heads = 2
# model_config.d_model=64
# model_config.context_length = 512

model_config = ModelConfig(
    d_model=128, 
    number_of_heads=2, 
    context_length=1024
    )

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
context = "Throughout the centuries, the Kenyan Coast has played host to many merchants and explorers. Among the cities that line the Kenyan coast is the City of Malindi. It has remained an important Swahili settlement since the 14th century and once rivalled Mombasa for dominance in the African Great Lakes region. Malindi has traditionally been a friendly port city for foreign powers. In 1414, the Chinese trader and explorer Zheng He representing the Ming Dynasty visited the East African coast on one of his last 'treasure voyages'. Malindi authorities welcomed the Portuguese explorer Vasco da Gama in 1498." 
question = "Where do many merchants and explorers go?"

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