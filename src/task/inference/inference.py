import torch

from src.infra.generation import prompt


class TextGenerator:

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
    def generate(
        self,
        prompt_text: str,
        max_new_tokens=300,
        temperature=0.7,
        top_p=None,
    ):

        prompt_tokens = torch.tensor(
            [self.tokenizer.encode(prompt_text)],
            dtype=torch.long,
            device=self.device,
        )

        generated_tokens = prompt(
            tokens=prompt_tokens,
            model=self.model,
            context_length=self.context_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return self.tokenizer.decode(
            generated_tokens[0].tolist()
        )