from pydoc import text

import torch


class TextClassifier:

    def __init__(
        self,
        model,
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
    def classify(
        self,
        prompt_text: str
    ):

        encoded = self.tokenizer.encode(prompt_text)

        tokens, padding_mask = self.tokenizer.pad(
            [encoded],
            max_length=self.context_length,
            )

        encoded_tokens = torch.tensor(
            tokens,
            dtype=torch.long,
            device=self.device,
        )

        encoded_mask = torch.tensor(
            padding_mask,
            dtype=torch.bool,
            device=self.device,
        )


        logits = self.model(
            encoded_tokens,
            padding_mask=encoded_mask,
        )

        prediction = torch.argmax(
            logits,
            dim=-1,
        )

        return prediction.item()
    