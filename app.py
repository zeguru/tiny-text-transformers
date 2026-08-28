import gradio as gr
import spaces
from pathlib import Path
from src.infra.tokenizer import CharacterTokenizer
from src.task.config import ModelConfig
from src.task.model import load_model
from src.task.inference import TextGenerator


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = Path("data/tiny_shakespear.txt")

MODEL_PATH = Path(
    "checkpoints/tiny-sentence-transformer.pt"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = ModelConfig()


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

text = DATA_PATH.read_text(
    encoding="utf-8",
)

tokenizer = CharacterTokenizer(text)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = load_model(
    checkpoint_path=MODEL_PATH,
    config=config,
    vocab_size=tokenizer.vocab_size,
)


# --------------------------------------------------
# Generator
# --------------------------------------------------

text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    context_length=config.context_length,
)


# --------------------------------------------------
# Gradio function
# --------------------------------------------------
@spaces.GPU
def generate_text(
    prompt,
    max_new_tokens,
    temperature,
):

    return text_generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# --------------------------------------------------
# Interface
# --------------------------------------------------

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            value="ROMEO:",
        ),
        gr.Slider(
            minimum=10,
            maximum=500,
            value=300,
            step=10,
            label="Max new tokens",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature",
        ),
    ],
    outputs=gr.Textbox(
        label="Generated text",
        lines=15,
    ),
    title="Tiny Sentence Transformer",
    description=(
        "A tiny character-level transformer "
        "trained on a Shakespeare corpus."
    ),
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )