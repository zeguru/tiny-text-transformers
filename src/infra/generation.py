import torch


# @torch.no_grad()        #do not compute gradients
# def generate(
#     tokens,
#     embedding,
#     model,
#     output_projection,
#     context_length,
#     max_new_tokens,
#     temperature=1.0,
# ):

#     if temperature <= 0:
#         raise ValueError("temperature must be greater than 0")

#     model.eval()

#     for _ in range(max_new_tokens):

#         context = tokens[:, -context_length:]

#         # ------------------------------------------
#         # Forward pass
#         # ------------------------------------------
#         x = embedding(context)
#         hidden = model(x)
#         logits = output_projection(hidden)

#         next_token_logits = logits[:, -1, :]

#         next_token_logits = (
#             next_token_logits / temperature
#         )

#         probabilities = torch.softmax(
#             next_token_logits,
#             dim=-1,
#         )

#         next_token = torch.multinomial(
#             probabilities,
#             num_samples=1,
#         )

#         tokens = torch.cat(
#             [tokens, next_token],
#             dim=1,
#         )

#     return tokens



@torch.no_grad()    #do not compute gradients
def generator(
    tokens,
    generator,
    context_length,
    max_new_tokens,
    temperature=1.0,
):

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    generator.eval()

    for _ in range(max_new_tokens):

        context = tokens[:, -context_length:]

        # ------------------------------------------
        # Forward pass
        # ------------------------------------------
       
        logits = generator(context)

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