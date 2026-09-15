from pathlib import Path

from src.task.config import ModelConfig
from src.task.model import load_model
from src.infra.tokenizer import CharacterTokenizer
from src.task.inference.inference import TextGenerator


# Runtime configs

checkpoint = "checkpoints/tiny_poet.pt"

TEXT_TOKENIZER_PATH = Path("checkpoints/shakespeare_tokenizer.json")
tokenizer = CharacterTokenizer.load(
    TEXT_TOKENIZER_PATH
    )

# Conf

generator_config = ModelConfig(context_length=128, d_model=128, d_ff=512, num_layers=4, number_of_heads=4)


# Load model/checkpoint from file 
model = load_model(
    checkpoint_path=checkpoint,
    config=generator_config,
    vocab_size=tokenizer.vocab_size,
)


# generator
text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    context_length=generator_config.context_length,
)


# do generate
output = text_generator.generate(
    prompt_text="ROMEO:",
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.8
)

print(output)