import torch
import torch.nn as nn
from torchgen import model
from src.infra.dataset.answering import QuestionAnsweringDataset
from torch.utils.data import DataLoader
from src.task.model import TinyAnsweringModel
from src.utils.device import get_device, move_to_device
from datetime import datetime
import time

# Teach 'em
def train(
    model: TinyAnsweringModel,
    train_dataset: QuestionAnsweringDataset,
    val_dataset: QuestionAnsweringDataset,
    training_config,
    ):

    optimal_device = get_device()   #cpu, mps, cuda
    


    print(f"Optimal device: {optimal_device}")

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

    model = model.to(optimal_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        )

    model.train()

    for epoch in range(training_config.num_steps):

        epoch_start = time.perf_counter()
        batch_times = []

        print(f"\nEpoch {epoch+1}/{training_config.num_steps}")

        for batch_idx, (tokens, padding_mask, start_position, end_position) in enumerate(train_loader):

            #print(f"\r\tProcessing batch : {batch_idx+1}", end=f" of {num_batches}  ", flush=True)

            tokens = tokens.to(optimal_device)
            padding_mask = padding_mask.to(optimal_device)
            start_position = start_position.to(optimal_device)
            end_position = end_position.to(optimal_device)

            device = next(model.parameters()).device
            #print(f"Actual device: {device}")

            batch_start = time.perf_counter()

            #===Investigate the actual device handling tensor operations
            # embedding = model.input_representation.token_embedding.embedding

            # print("Model parameter:", next(model.parameters()).device)
            # print("Tokens:", tokens.device, tokens.dtype)
            # print("Embedding weight device:", embedding.weight.device)
            # print("Embedding weight is_meta:", embedding.weight.is_meta)
            # print("Embedding weight shape:", embedding.weight.shape)
            #===

            start_logits, end_logits = model(tokens,padding_mask)

            start_position = start_position.long()
            end_position = end_position.long() -1       #bounds bug
        
            start_loss = loss_function(start_logits,start_position)
            end_loss = loss_function(end_logits,end_position)
            
            loss = start_loss + end_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Wait for MPS work to finish before stopping the timer
            if device.type == "mps":
                torch.mps.synchronize()

            batch_elapsed = time.perf_counter() - batch_start
            batch_times.append(batch_elapsed)

            print(f"\r\tProcessed batch : {batch_idx+1}", end=f" of {num_batches}  Time: {batch_elapsed:.4f}s", flush=True)


        epoch_elapsed = time.perf_counter() - epoch_start

        print(f"\n\tEpoch time: {epoch_elapsed:.2f}s")
        print(f"\tAverage batch time: {sum(batch_times) / len(batch_times):.4f}s")

        #poor-man's eval
        if epoch % training_config.eval_interval == 0:
            print(f"\t*Loss : {loss}")

    return model
