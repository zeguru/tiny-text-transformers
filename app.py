import spaces
import gradio as gr

from pathlib import Path
from src.infra.tokenizer import CharacterTokenizer
from src.task.config import ModelConfig

from src.task.model import load_model, load_classifier_model

from src.task.inference.inference import TextGenerator
from src.task.inference.classify import TextClassifier

# Paths
# DATA_PATH = Path("data/tiny_shakespear.txt")
GENERATOR_MODEL_PATH = Path("checkpoints/tiny_poet.pt")

CLASSIFIER_MODEL_PATH = Path(
    "checkpoints/mh_tiny_text_classifier.pt"
    )

CLASSIFIER_TOKENIZER_PATH = Path(
    "checkpoints/ag_news_tokenizer.json"
    )


TEXT_TOKENIZER_PATH = Path("checkpoints/shakespeare_tokenizer.json")
tokenizer = CharacterTokenizer.load(TEXT_TOKENIZER_PATH)

generator_config = ModelConfig(context_length=128, d_model=128, d_ff=512, num_layers=4, number_of_heads=4)

model = load_model(
    checkpoint_path=GENERATOR_MODEL_PATH,
    config=generator_config,
    vocab_size=tokenizer.vocab_size,
    )

text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    context_length=generator_config.context_length,
    )

# Text Generation
@spaces.GPU
def generate_text(
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    ):

    return text_generator.generate(
        prompt_text=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
        )

# B. Classifier
classifier_tokenizer = CharacterTokenizer.load(CLASSIFIER_TOKENIZER_PATH)

classifier_config = ModelConfig(context_length=384, d_model=64, d_ff=256, num_layers=2)

classifier_model = load_classifier_model(
    checkpoint_path=CLASSIFIER_MODEL_PATH,
    config=classifier_config,
    vocab_size=classifier_tokenizer.vocab_size,
    )

text_classifier = TextClassifier(
    model=classifier_model,
    tokenizer=classifier_tokenizer,
    context_length=classifier_config.context_length,
    )

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


@spaces.GPU
def classify_text(text):
    prediction = text_classifier.classify(text)

    return CLASS_NAMES[prediction]



with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Tiny Text Transformers

        A set of tiny character-level Transformers built from first principles using Pytorch,
        demonstrated on text Generation and text Classification.

        Generation was trained on Shakespeare txt, Classification was trained on AG News (World, Sport, Business and Sci/Tech)

        """
    )

    with gr.Tab("Text Generation"):
        gr.Interface(
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
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Top P",
                ),
            ],
            outputs=gr.Textbox(
                label="Generated text",
                lines=15,
            ),
        )

    with gr.Tab("Text Classification"):
        gr.Interface(
            fn=classify_text,
            inputs=gr.Textbox(
                label="News text",
                value="The newest Mac mini was announced on August to launch in stores on September",
                lines=5,
            ),
            outputs=gr.Textbox(
                label="Predicted category",
            ),
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        )