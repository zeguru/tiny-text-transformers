import torch


@torch.no_grad()    #do not compute gradients
def prompt(
    tokens,
    model,
    context_length,
    max_new_tokens,
    temperature=1.0,
):

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    model.eval()

    for _ in range(max_new_tokens):

        context = tokens[:, -context_length:]

        # ------------------------------------------
        # Forward pass
        # ------------------------------------------
       
        logits = model(context)

        next_token_logits = logits[:, -1, :]

        next_token_logits = (
            next_token_logits / temperature
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    return tokens