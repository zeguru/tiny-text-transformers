import torch
from pathlib import Path
from src.infra.tokenizer import CharacterTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


print("Starting Tokenizer Checks....")

TEXT_TOKENIZER_PATH = Path("checkpoints/shakespeare-tokenizer.json")

tokenizer = CharacterTokenizer("hello world")
tokenizer.save(TEXT_TOKENIZER_PATH)
loaded = CharacterTokenizer.load(TEXT_TOKENIZER_PATH)

assert tokenizer.stoi == loaded.stoi
assert tokenizer.itos == loaded.itos
assert tokenizer.vocab_size == loaded.vocab_size

text = "hello world"

assert tokenizer.decode(
    loaded.encode(text)
) == text

print("Tokenizer Checks Complete")
