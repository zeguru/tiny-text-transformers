import torch
import torch.nn as nn
from torchgen import model
from src.infra.dataset.answering import QuestionAnsweringDataset
from torch.utils.data import DataLoader
from src.task.model import TinyAnsweringModel

# Teach 'em
def train(
    model: TinyAnsweringModel,
    train_dataset: QuestionAnsweringDataset,
    val_dataset: QuestionAnsweringDataset,
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

    model.train()

    for epoch in range(training_config.num_steps):

        print(f"\nEpoch {epoch+1}/{training_config.num_steps}")

        #for tokens, padding_mask, start_position, end_position in train_loader:
        for batch_idx, (tokens, padding_mask, start_position, end_position) in enumerate(train_loader):

            print(f"\r\tProcessing batch : {batch_idx+1}", end=f" of {num_batches}  ", flush=True)

            start_logits, end_logits = model(tokens,padding_mask)

            start_position = start_position.long()
            end_position = end_position.long() -1       #bounds bug
        
            start_loss = loss_function(start_logits,start_position)
            end_loss = loss_function(end_logits,end_position)
            
            loss = start_loss + end_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #poor-man's eval
        if epoch % training_config.eval_interval == 0:
            print(f"\tLoss : {loss}")

    return model
