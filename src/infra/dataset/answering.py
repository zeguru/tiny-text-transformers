import torch
from torch.utils.data import Dataset
from src.infra.tokenizer import CharacterTokenizer


class QuestionAnsweringDataset(Dataset):

    """Dataset for QnA. Extractive AI

    A token sequence is composed of question, context, answer_start and answer end.

    For each:
        input:  The question and the context containing the answer
            <CLS>question<SEP>context<SEP>

        target:  Character positions that locate the answer in the input
            answer start, answer end. Whatch-out for 'out-of-bounds' bugs !

    The model must learn that the answer appears between the <SEP> markers 
            
    """
        
    def __init__(
        self,
        records,
        tokenizer: CharacterTokenizer,
        max_length,
        ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for record in records:
            question = record["question"]
            context = record["context"]
            answer_text = record["answer"]
            answer_start = record["answer_start"]
            answer_end = answer_start + len(answer_text)

            if context[answer_start:answer_end] != answer_text: #Do not allow wrong answers
                continue

            # Build the model input.
            input_text = (
                tokenizer.CLS_TOKEN
                + question
                + tokenizer.SEP_TOKEN
                + context
                + tokenizer.SEP_TOKEN
                )

            input_ids = tokenizer.encode(input_text)

            question_ids = tokenizer.encode(
                tokenizer.CLS_TOKEN + question
                )
            separator_ids = tokenizer.encode(
                tokenizer.SEP_TOKEN
                )

            context_offset = len(question_ids) + len(separator_ids)

            start_position = context_offset + answer_start
            end_position = context_offset + answer_end
            
            if end_position > max_length:   # Reject answers outside the bounds
                continue

            # Pad returns ids of the surviving inputs and padded positions, if any
            self.padded_ids, self.masked_ids = tokenizer.pad(
                [input_ids],        #Replaced [truncated]
                max_length=max_length,
                )

            #We might need to later explore streaming datasets, 
            #Loading all the records in memory is not cool at all
            self.data.append({
                "input_tokens": torch.tensor(
                    self.padded_ids[0], dtype=torch.long
                ),
                "padding_mask": torch.tensor(
                    self.masked_ids[0], dtype=torch.bool
                ),
                "start_position": torch.tensor(
                    start_position, dtype=torch.long
                ),
                "end_position": torch.tensor(
                    end_position, dtype=torch.long
                ),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        return (
            sample["input_tokens"],
            sample["padding_mask"],
            sample["start_position"],
            sample["end_position"],
        )