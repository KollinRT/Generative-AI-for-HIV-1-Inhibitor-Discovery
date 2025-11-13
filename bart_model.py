import torch
import pandas as pd
import random
import selfies as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer


def collate_fn(batch):
    # If the dataset returns dictionaries (mode='finetune')
    if isinstance(batch[0], dict):
        # Padding input IDs
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=0
        )

        # Stack IC50 and inhibition site data
        ic50 = torch.stack([item["IC50"] for item in batch])
        inhibition_site = torch.stack([item["inhibition_site"] for item in batch])

        return {
            "input_ids": input_ids,
            "IC50": ic50,
            "inhibition_site": inhibition_site,
        }
    else:  # Pretraining mode, where only input_ids are expected
        input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
        return input_ids


class SelfiesDataset(Dataset):
    def __init__(self, csv_file, tokenizer_path, mode="pretrain"):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        self.data = pd.read_csv(csv_file)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.mode = mode  # Options are 'pretrain' or 'finetune'

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        selfies_string = self.data.iloc[idx]["selfies"]  # selfies not SELFIES
        encoded = self.tokenizer.encode(selfies_string)

        if self.mode == "pretrain":
            return torch.tensor(encoded.ids, dtype=torch.long)
        elif self.mode == "finetune":
            IC50 = self.data.iloc[idx]["IC50"]
            inhibition_site = self.data.iloc[idx]["site_name"]
            inhibition_encoded = self.encode_inhibition_site(inhibition_site)

            return {
                "input_ids": torch.tensor(encoded.ids, dtype=torch.long),
                "IC50": torch.tensor([IC50], dtype=torch.float),
                "inhibition_site": torch.tensor([inhibition_encoded], dtype=torch.long),
            }

    def encode_inhibition_site(self, inhibition_site):
        """Encodes the inhibition site after normalizing string to prevent matching errors."""
        inhibition_site = (
            inhibition_site.strip().upper()
        )  # Normalize by trimming spaces and converting to uppercase
        if "RVP" in inhibition_site:
            return 0
        elif "RVE" in inhibition_site:
            return 1
        return -1  # Return -1 for cases where neither RVP nor RVE is found


class SelfiesTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab  # Assumes <start>, <end>, <pad>, <unk> are part of the vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode(self, selfies_string):
        symbols = ["<start>"] + sf.split_selfies(selfies_string) + ["<end>"]
        token_ids = [self.vocab.get(symbol, self.vocab["<unk>"]) for symbol in symbols]
        return token_ids

    def decode(self, token_ids):
        symbols = [self.inv_vocab[id] for id in token_ids if id in self.inv_vocab]
        selfies_string = "".join(symbols)
        return (
            selfies_string.replace("<start>", "")
            .replace("<end>", "")
            .replace("<pad>", "")
        )


def mutate_selfies(selfies_string, mutation_rate=0.1):
    symbols = sf.split_selfies(
        selfies_string
    )  # Split the SELFIES into individual symbols
    mutated_symbols = []
    for symbol in symbols:
        if random.random() < mutation_rate:
            # Randomly replace a symbol with another
            all_symbols = list(sf.get_semantic_robust_alphabet())
            new_symbol = random.choice(all_symbols)
            mutated_symbols.append(new_symbol)
        else:
            mutated_symbols.append(symbol)
    mutated_selfies = "".join(mutated_symbols)  # Directly concatenate the symbols
    return mutated_selfies


def build_selfies_vocab(selfies_list):
    unique_symbols = set()
    # Ensure that selfies is a string before processing
    for selfies in filter(lambda x: isinstance(x, str), selfies_list):
        symbols = sf.split_selfies(selfies)
        unique_symbols.update(symbols)
    return {symbol: i for i, symbol in enumerate(unique_symbols)}