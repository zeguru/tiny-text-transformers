import torch


@torch.no_grad()    #do not compute gradients
def prompt(
    tokens,
    model,
    context_length,
    max_new_tokens,
    temperature=1.0,
    top_p=None
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

        if top_p is not None:
            probabilities = get_top_p(top_p, next_token_logits)
        else:
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
                )

        next_token = torch.multinomial(     #watermark my be done here 
            probabilities,
            num_samples=1,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    return tokens


def get_top_p(top_p,next_logits):

    probabilities = torch.softmax(
        next_logits,
        dim=-1,
    )

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    remove = cumulative_probabilities > top_p

    # Keep the first token that crosses the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_probabilities = sorted_probabilities.masked_fill(
        remove,
        0.0,
        )

    probabilities = torch.zeros_like(probabilities)

    probabilities.scatter_(
        -1,
        sorted_indices,
        sorted_probabilities,
        )

    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True,
        )

    return probabilities