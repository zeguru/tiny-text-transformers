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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Training Batch size: {training_config.batch_size}")
    print(f"Batches per epoch: {num_batches}")
    print(f"Total batches: {num_batches * training_config.num_steps}")

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Validation Batch size: {training_config.eval_batches}")


    # logger.log(f"Train examples: {len(train_dataset):,}")
    # logger.log(f"Val examples:   {len(val_dataset):,}")
    # logger.log(f"Test examples:  {len(test_dataset):,}")

    loss_function = nn.CrossEntropyLoss()

    model = model.to(optimal_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        )




    history = {
        "train_loss": [],
        "val_loss": [],
        "start_accuracy": [],
        "end_accuracy": [],
        "exact_match": [],
    }
    
    for epoch in range(training_config.num_steps):

        model.train()
    
        epoch_loss = 0.0
        total_examples = 0

        epoch_start = time.perf_counter()
        batch_times = []


        print(f"\nEpoch {epoch+1}/{training_config.num_steps}")

        for batch_idx, (tokens, padding_mask, start_position, end_position) in enumerate(train_loader):

            #print(f"\r\tProcessing batch : {batch_idx+1}", end=f" of {num_batches}  ", flush=True)

            tokens = tokens.to(optimal_device)
            padding_mask = padding_mask.to(optimal_device)
            start_position = start_position.to(optimal_device)
            end_position = end_position.to(optimal_device)

            batch_start = time.perf_counter()

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
            device = next(model.parameters()).device
            #print(f"Actual device: {device}")
            if device.type == "mps":
                torch.mps.synchronize()

            batch_elapsed = time.perf_counter() - batch_start
            batch_times.append(batch_elapsed)

            print(f"\r\tProcessed batch : {batch_idx+1}", end=f" of {num_batches}  Time: {batch_elapsed:.4f}s", flush=True)

            loss = start_loss + end_loss
            batch_size_actual = tokens.size(0)
            epoch_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

        epoch_elapsed = time.perf_counter() - epoch_start

        print(f"\n\tEpoch time: {epoch_elapsed:.2f}s")
        print(f"\tAverage batch time: {sum(batch_times) / len(batch_times):.4f}s")

        avg_train_loss = epoch_loss / total_examples
        print(f"\tTraining Loss: {avg_train_loss:.4f}")
        history["train_loss"].append(avg_train_loss)

        #new eval
        if (epoch + 1) % training_config.eval_interval == 0:

            metrics = evaluate(
                model=model,
                val_dataset=val_dataset,
                batch_size=training_config.eval_batches,
                device=optimal_device,
            )

            print(f"\n\tValidation Loss: {metrics['loss']:.4f}")
            print(f"\t\tAnswer Start Accuracy: {metrics['start_accuracy']:.2%}")
            print(f"\t\tAnswer End Accuracy: {metrics['end_accuracy']:.2%}")
            print(f"\t\tAnswer Exact Match (start and end): {metrics['exact_match']:.2%}")

            history["val_loss"].append(metrics["loss"])
            history["start_accuracy"].append(metrics["start_accuracy"])
            history["end_accuracy"].append(metrics["end_accuracy"])
            history["exact_match"].append(metrics["exact_match"])

            model.train()

    return model,history

def evaluate(
    model: TinyAnsweringModel,
    val_dataset: QuestionAnsweringDataset,
    batch_size: int,
    device: torch.device,
):
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_function = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_examples = 0

    correct_start = 0
    correct_end = 0
    correct_span = 0

    with torch.no_grad():

        for tokens, padding_mask, start_position, end_position in val_loader:

            tokens = tokens.to(device)
            padding_mask = padding_mask.to(device)
            start_position = start_position.to(device).long()
            end_position = end_position.to(device).long() - 1

            start_logits, end_logits = model(tokens, padding_mask)

            start_loss = loss_function(start_logits, start_position)
            end_loss = loss_function(end_logits, end_position)

            loss = start_loss + end_loss

            batch_size_actual = tokens.size(0)

            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

            predicted_start = start_logits.argmax(dim=1)
            predicted_end = end_logits.argmax(dim=1)

            correct_start += (predicted_start == start_position).sum().item()
            correct_end += (predicted_end == end_position).sum().item()

            correct_span += (
                (predicted_start == start_position) &
                (predicted_end == end_position)
            ).sum().item()

    avg_loss = total_loss / total_examples

    start_accuracy = correct_start / total_examples
    end_accuracy = correct_end / total_examples
    exact_match = correct_span / total_examples

    return {
        "loss": avg_loss,
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "exact_match": exact_match,
    }