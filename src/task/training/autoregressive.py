import torch
import torch.nn as nn

from src.infra.dataset.autoregressive import AutoRegressiveDataset
from src.infra.dataset.util import get_batch



# Teach 'em
def train(
    model,
    train_dataset: AutoRegressiveDataset,
    val_dataset: AutoRegressiveDataset,
    training_config,
    ):

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        )

    model.train()

    for step in range(training_config.num_steps):

        if step % training_config.eval_interval == 0:

            train_loss = evaluate(
                model=model,
                dataset=train_dataset,
                training_config=training_config,
                )

            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                training_config=training_config,
                )

            print(
                f"step {step:4d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f}"
                )

        inputs, targets = get_batch(
            dataset=train_dataset,
            batch_size=training_config.batch_size,
            )

        logits = model(inputs)

        loss = loss_function(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


@torch.no_grad()    #do not calculate gradients
def evaluate(
    model,
    dataset: AutoRegressiveDataset,
    training_config,
    ):

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    losses = []

    for _ in range(training_config.eval_batches):

        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=training_config.batch_size,
            )

        logits = model(inputs)

        loss = loss_function(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


