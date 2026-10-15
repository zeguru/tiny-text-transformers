from pydoc import text

import torch
from src.task.model import TinyTransformerLM, TinyClassifier, TinyAnsweringModel

class TextExtractor:

    def __init__(
        self,
        model: TinyAnsweringModel,
        tokenizer,
        context_length,
        device=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(
        self,
        context_text: str,  #context containing the answer
        question_text: str  #the actual question 
        ):

        input_text = (
                self.tokenizer.CLS_TOKEN
                + question_text
                + self.tokenizer.SEP_TOKEN
                + context_text
                + self.tokenizer.SEP_TOKEN
                )
        input_ids = self.tokenizer.encode(input_text)

        padded_ids, mask_ids = self.tokenizer.pad(
            [input_ids],
            max_length=self.context_length
            )

        token_tensor = torch.tensor(
            padded_ids[0],
            dtype=torch.long,
            device=self.device,
            )

        padding_mask = torch.tensor(
            mask_ids,
            dtype=torch.bool,
            device=self.device,
            )

        #Typically -> embedding -> transformer -> start,end logits
        #Forward pass on the model is refusing :-(
        start_logits, end_logits = self.model( 
            tokens=token_tensor.unsqueeze(0),
            padding_mask=padding_mask
            )


        # Remove batch dimension: [max_length]
        start_logits = start_logits[0]
        end_logits = end_logits[0]
        
        start_predictions = torch.argmax(
            start_logits,
            dim=-1,
            )
    
        end_predictions = torch.argmax(
            end_logits,
            dim=-1,
            )
        
        return token_tensor, start_predictions.item(), end_predictions.item()
    