from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference import TextGenerator


# ------------------------------------------
# Load corpus
# ------------------------------------------

text = Path(
    "data/tiny_shakespear.txt"
).read_text(
    encoding="utf-8"
)


# ------------------------------------------
# Tokenizer
# ------------------------------------------

tokenizer = CharacterTokenizer(text)


# ------------------------------------------
# Configuration
# ------------------------------------------

config = ModelConfig()


# ------------------------------------------
# Load trained model
# ------------------------------------------

model = load_model(
    checkpoint_path="checkpoints/tiny-sentence-transformer.pt",
    config=config,
    vocab_size=tokenizer.vocab_size,
)


# ------------------------------------------
# Generator
# ------------------------------------------

text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    context_length=config.context_length,
)


# ------------------------------------------
# Generate
# ------------------------------------------

output = text_generator.generate(
    prompt="ROMEO:",
    max_new_tokens=300,
    temperature=0.7,
)


print(output)