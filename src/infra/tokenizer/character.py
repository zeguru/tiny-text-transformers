import json
from pathlib import Path

# Treats each character as a token 
class CharacterTokenizer:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"         #classification token
    SEP_TOKEN = "<SEP>"         #separator 

    def __init__(self, text):
        characters = sorted(set(text))      #sorts + removes duplicates

        self.stoi = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
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

    @property
    def cls_token_id(self):
        return self.stoi[self.CLS_TOKEN]

    @property
    def sep_token_id(self):
        return self.stoi[self.SEP_TOKEN]

    # Original encode before CLS and SEP tokens
    # def encode(self, text):
    #     return [
    #         self.stoi.get(
    #             character,
    #             self.unk_token_id,
    #         )
    #         for character in text
    #     ]

    def encode(self, text):
        tokens = []
        i = 0

        while i < len(text):

            if text.startswith(self.CLS_TOKEN, i):  #start checking from here 
                tokens.append(self.cls_token_id)
                i += len(self.CLS_TOKEN)        #skip the length of the token

            elif text.startswith(self.SEP_TOKEN, i):
                tokens.append(self.sep_token_id)
                i += len(self.SEP_TOKEN)        #skip the length of the token

            else:
                tokens.append(
                    self.stoi.get(
                        text[i],
                        self.unk_token_id,
                        )
                    )
                i += 1

        return tokens       #actually indices in the token space
    
    def decode(self, token_ids):
        return "".join(
            self.itos[token_id]
            for token_id in token_ids
            if token_id != self.pad_token_id
            )
    
    
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
            "version": 2,
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