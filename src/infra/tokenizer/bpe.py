
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"         #classification token
    SEP_TOKEN = "<SEP>"         #separator 
    # BOS_TOKEN = "[BOS]"
    # EOS_TOKEN = "[EOS]"

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.pad_token_id = tokenizer.token_to_id(
            self.PAD_TOKEN
        )
        self.unk_token_id = tokenizer.token_to_id(
            self.UNK_TOKEN
        )
        self.cls_token_id = tokenizer.token_to_id(
            self.CLS_TOKEN
        )
        self.sep_token_id = tokenizer.token_to_id(
            self.SEP_TOKEN
        )

        if any(
            token_id is None
            for token_id in (
                self.pad_token_id,
                self.unk_token_id,
                self.cls_token_id,
                self.sep_token_id,
                )
            ):
            raise ValueError(
                "Tokenizer is missing one or more required "
                "special tokens."
            )

    @classmethod
    def train(
        cls,
        texts,
        vocab_size=2_000,
    ):
        tokenizer = Tokenizer(
            BPE(unk_token=cls.UNK_TOKEN)
            )

        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
            )
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                cls.PAD_TOKEN,
                cls.UNK_TOKEN,
                cls.CLS_TOKEN,
                cls.SEP_TOKEN,
                ],
            initial_alphabet=ByteLevel.alphabet(),
            )

        tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
            )

        return cls(tokenizer)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        # Do not render padding or structural special tokens
        # as part of ordinary decoded text.
        token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in {
                self.pad_token_id,
                self.cls_token_id,
                self.sep_token_id,
                }
            ]

        return self.tokenizer.decode(token_ids)

    def pad(self, sequences, max_length=None):
        if not sequences:
            raise ValueError(
                "Cannot pad an empty collection of sequences."
            )

        if max_length is None:
            max_length = max(
                len(sequence)
                for sequence in sequences
            )

        if max_length < 1:
            raise ValueError("max_length must be positive.")

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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
            )

        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path):
        tokenizer = Tokenizer.from_file(
            str(Path(path))
            )

        return cls(tokenizer)
