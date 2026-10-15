import spaces
import gradio as gr

from pathlib import Path
from src.infra.tokenizer import CharacterTokenizer
from src.task.config import ModelConfig

from src.task.model import load_model

from src.task.inference.inference import TextGenerator
from src.task.inference.classify import TextClassifier
from src.task.inference.extract import TextExtractor

# Paths
# DATA_PATH = Path("data/tiny_shakespear.txt")
GENERATOR_MODEL_PATH = Path("checkpoints/tiny_poet.pt")
CLASSIFIER_MODEL_PATH = Path("checkpoints/mh_tiny_text_classifier.pt")
ANSWERING_MODEL_PATH = Path("checkpoints/tiny_answering.pt")


CLASSIFIER_TOKENIZER_PATH = Path("checkpoints/ag_news_tokenizer.json")
TEXT_TOKENIZER_PATH = Path("checkpoints/shakespeare_tokenizer.json")
QA_TOKENIZER_PATH = Path("checkpoints/qa_tokenizer.json")

#A. GENERATOR
#==========================
generator_tokenizer = CharacterTokenizer.load(TEXT_TOKENIZER_PATH)

generator_config = ModelConfig(context_length=128, d_model=128, d_ff=512, num_layers=4, number_of_heads=4)

generator_model = load_model(
    checkpoint_path=GENERATOR_MODEL_PATH,
    config=generator_config,
    vocab_size=generator_tokenizer.vocab_size,
    type="GENERATOR"
    )

text_generator = TextGenerator(
    model=generator_model,
    tokenizer=generator_tokenizer,
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

# B. CLASSIFIER
#==========================

classifier_tokenizer = CharacterTokenizer.load(CLASSIFIER_TOKENIZER_PATH)

classifier_config = ModelConfig(context_length=384, d_model=64, d_ff=256, num_layers=2)

classifier_model = load_model(
    checkpoint_path=CLASSIFIER_MODEL_PATH,
    config=classifier_config,
    vocab_size=classifier_tokenizer.vocab_size,
    type="CLASSIFIER"
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

#C. QUESTION ANSWERING
qa_tokenizer = CharacterTokenizer.load(QA_TOKENIZER_PATH)

qa_config = ModelConfig(context_length=512, d_model=64, d_ff=256, num_layers=2)

answering_model = load_model(
    checkpoint_path=ANSWERING_MODEL_PATH,
    config=qa_config,
    vocab_size=qa_tokenizer.vocab_size,
    type="ANSWERING"
    )

text_extractor = TextExtractor(
    model=answering_model,
    tokenizer=qa_tokenizer,
    context_length=qa_config.context_length,
    )


@spaces.GPU
def ask_question(
    context,
    question
    ):

    answer_text = ""

    token_tensor, predicted_start, predicted_end = text_extractor.extract(
        context_text=context,
        question_text=question
        )
    if (predicted_start >= predicted_end):
        answer_text = "Invalid Answer Found. OOPS"
    else:
        answer_ids = token_tensor[predicted_start:predicted_end + 1]
        answer_text = qa_tokenizer.decode(answer_ids.tolist())

    return answer_text


#USER INTERFACE
#==========================

#. 
with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Tiny Text Transformers

        A set of tiny character-level Transformers built from first principles using Pytorch.
        POCs:- Generation, Classification and Question Answering (extractive).

        Generation was trained on Shakespeare text and Tiny Stories, 
        Classification was trained on AG News (World, Sport, Business and Sci/Tech),
        Question Answering was trained on a subset of SQuAD v1.1 dataset (kenya).

        NOTE: These are not general purpose models and will only return useful responses within the context of the datasets above. 

        Goals:  
            - Build capable but highly specialized models
            - Explore limits of micro intelligence
            - Fluency in tranformer design (personal)
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

    with gr.Tab("Question Answering"):
        gr.Interface(
            fn=ask_question,
            inputs=[
                gr.Textbox(
                    label="Context",
                    value="Kenya (/ˈkɛnjə/; locally [ˈkɛɲa] ( listen)), officially the Republic of Kenya, is a country in Africa and a founding member of the East African Community (EAC). Its capital and largest city is Nairobi. Kenya's territory lies on the equator and overlies the East African Rift covering a diverse and expansive terrain that extends roughly from Lake Victoria to Lake Turkana (formerly called Lake Rudolf) and further south-east to the Indian Ocean. It is bordered by Tanzania to the south, Uganda to the west, South Sudan to the north-west, Ethiopia to the north and Somalia to the north-east. Kenya covers 581,309 km2 (224,445 sq mi), and had a population of approximately 45 million people in July 2014.",
                    lines=5,
                    ),
                gr.Textbox(
                    label="Question",
                    value="what is the capital of kenya?",
                    )
                ],
            outputs=gr.Textbox(
                label="Predicted answer",
            ),
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        )