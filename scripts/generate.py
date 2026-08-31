from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference import TextGenerator


# Runtime configs
corpus = "data/tiny_shakespear.txt"
checkpoint = "checkpoints/tiny-text-transformer-0.2.pt"

text = Path(corpus).read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)


# Conf

config = ModelConfig()


# Load model/checkpoint from file 
model = load_model(
    checkpoint_path=checkpoint,
    config=config,
    vocab_size=tokenizer.vocab_size,
)


# generator
text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    context_length=config.context_length,
)


# do gemerate
output = text_generator.generate(
    prompt="ROMEO:",
    max_new_tokens=300,
    temperature=0.7,
)

print(output)