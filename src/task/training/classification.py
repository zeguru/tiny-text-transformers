import torch
import torch.nn as nn
from torchgen import model
from src.infra.dataset.classification import TextClassificationDataset
from torch.utils.data import DataLoader
from datetime import datetime


# Teach 'em
def train(
    model,
    train_dataset: TextClassificationDataset,
    val_dataset: TextClassificationDataset,
    training_config,
    ):

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        )

    num_batches = len(train_loader)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Batches per epoch: {num_batches}")
    print(f"Total batches: {num_batches * training_config.num_steps}")


    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        )

    for epoch in range(training_config.num_steps):

        print(f"Epoch {epoch+1}/{training_config.num_steps}")

        if epoch % training_config.eval_interval == 0:

            train_loss, train_accuracy = evaluate(
                model,
                train_dataset,
                training_config,
                )

            val_loss, val_accuracy = evaluate(
                model,
                val_dataset,
                training_config,
                )

            print(
                f"Train loss: {train_loss:.4f} "
                f"Train acc: {train_accuracy:.2%} "
                f"Val loss: {val_loss:.4f} "
                f"Val acc: {val_accuracy:.2%}"
                )

        for tokens, padding_mask, labels in train_loader:      #equivalent to get_batch
            model.train()
            logits = model(
                tokens,
                padding_mask=padding_mask,
                )
        
            loss = loss_function(
                logits,
                labels,
                )
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            #return loss.item()      # return loss.item() vs return model

    return model


@torch.no_grad()
def evaluate(
    model,
    dataset: TextClassificationDataset,
    training_config,
):
    eval_loader = DataLoader(
        dataset,
        batch_size=training_config.eval_batches,
        shuffle=False,
    )

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for tokens, padding_mask, labels in eval_loader:

        logits = model(
            tokens,
            padding_mask=padding_mask,
        )

        loss = loss_function(
            logits,
            labels,
        )

        losses.append(loss.item())

        predictions = torch.argmax(
            logits,
            dim=-1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    model.train()

    average_loss = sum(losses) / len(losses)
    accuracy = correct / total

    return average_loss, accuracy