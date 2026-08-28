import torch

from src.infra.generation import generator


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
        prompt,
        max_new_tokens=300,
        temperature=0.7,
    ):

        prompt_tokens = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=self.device,
        )

        generated_tokens = generator(
            tokens=prompt_tokens,
            generator=self.model,
            context_length=self.context_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return self.tokenizer.decode(
            generated_tokens[0].tolist()
        )