import json
from pathlib import Path

# Treats each character is a token 
class CharacterTokenizer:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, text):
        characters = sorted(set(text))

        self.stoi = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        for character in characters:
            self.stoi[character] = len(self.stoi)

        self.itos = {
            index: token
            for token, index in self.stoi.items()
        }

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_token_id(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def unk_token_id(self):
        return self.stoi[self.UNK_TOKEN]

    def encode(self, text):
        return [
            self.stoi.get(
                character,
                self.unk_token_id,
            )
            for character in text
        ]

    def decode(self, token_ids):
        return "".join(
            self.itos[token_id]
            for token_id in token_ids
            if token_id != self.pad_token_id
        )
    
    #do not assume tokenids are python integerser
    # def decode(self, token_ids):
    #     return "".join(
    #         self.itos[int(token_id)]
    #         for token_id in token_ids
    #         if int(token_id) != self.pad_token_id
    #     )
    
    def pad(
            self,
            sequences,
            max_length=None,
            ):
        
        if max_length is None:
            max_length = max(
                len(sequence)
                for sequence in sequences
            )

        padded = []
        masks = []

        for sequence in sequences:

            sequence = sequence[:max_length]

            padding_length = (
                max_length - len(sequence)
            )

            padded_sequence = (
                sequence
                + [self.pad_token_id] * padding_length
            )

            mask = (
                [1] * len(sequence)
                + [0] * padding_length
            )

            padded.append(padded_sequence)
            masks.append(mask)

        return padded, masks


    #Save the tokenizer
    def save(self, path):
        path = Path(path)

        data = {
            "version": 1,
            "stoi": self.stoi,
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

#Load saved
    @classmethod
    def load(cls, path):
        path = Path(path)

        data = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        tokenizer = cls.__new__(cls)

        tokenizer.stoi = data["stoi"]

        tokenizer.itos = {
            index: token
            for token, index in tokenizer.stoi.items()
        }

        return tokenizer