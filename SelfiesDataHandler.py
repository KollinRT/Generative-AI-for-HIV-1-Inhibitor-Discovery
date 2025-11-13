import torch
import random
import selfies as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

import dask.dataframe as dd


def collate_fn_pre(batch):
    # Pretraining mode: items only have 'input_ids'
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = (
        input_ids != 0
    ).long()  # Create attention mask (1 for tokens, 0 for padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn_fine(batch):
    """Collate function for DataLoader to pad sequences dynamically."""

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad sequences to the max length in the batch
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )  # Updated to the right padding value
    attention_masks_padded = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )  # 0 for padding in mask

    batch_dict = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
    }

    # If finetune mode, include additional targets
    if "IC50" in batch[0]:
        batch_dict["IC50"] = torch.stack([item["IC50"] for item in batch])
        batch_dict["inhibition_site"] = torch.stack(
            [item["inhibition_site"] for item in batch]
        )

    return batch_dict


def collate_fn(batch, mode="fine"):
    input_ids = [item["input_ids"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_masks_padded = (
        pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        if mode == "fine"
        else (input_ids_padded != 0).long()
    )

    batch_dict = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
    }

    # Add labels for pretrain mode (or fine mode if you have labels)
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 to ignore in loss
        batch_dict["labels"] = labels_padded

    if mode == "fine":
        if "IC50" in batch[0]:
            batch_dict["IC50"] = torch.stack([item["IC50"] for item in batch])
        if "inhibition_site" in batch[0]:
            batch_dict["inhibition_site"] = torch.stack(
                [item["inhibition_site"] for item in batch]
            )

    return batch_dict

class SelfiesDatasetRound2(Dataset):
    # def __init__(self, csv_file, tokenizer_path, mode='pretrain'):
    def __init__(self, dataframe, tokenizer, mode='pretrain'):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        # self.data = pd.read_csv(csv_file)
        # self.data = dataframe
        self.dataframe = dataframe.reset_index(drop=True)  # Ensure indices are 0,1,2,...


        # print(f"CSV columns: {self.data.columns.tolist()}")  # Debugging print statement
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        self.mode = mode  # Options are 'pretrain' or 'finetune'

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        selfies_string = str(self.dataframe.iloc[idx]['selfies'])  # force cast to str
        #selfies_string = self.dataframe.iloc[idx]['selfies']
        #print("selfies_string:", selfies_string)
        # Correctly tokenize SELFIES using semantic splitting
        tokens = list(sf.split_selfies(selfies_string))  # ['[C]', '[C]', '[O]']
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

        #print("input_ids:", input_ids)
        # Pad/truncate to max_length (e.g., 256)
        # max_length = 256
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        #

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return sample

    def encode_inhibition_site(self, inhibition_site):
        """Encodes the inhibition site after normalizing string to prevent matching errors."""
        inhibition_site = inhibition_site.strip().upper()  # Normalize by trimming spaces and converting to uppercase
        if 'RVP' in inhibition_site:
            return 0
        elif 'RVE' in inhibition_site:
            return 1
        return -1  # Return -1 for cases where neither RVP nor RVE is found


class SelfiesDataset(Dataset):
    def __init__(self, dataframe, tokenizer, mode="pretrain", mask_prob=0.15):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        # self.dataframe = dataframe.reset_index(drop=True)  # Ensure indices are 0,1,2,...
        self.dataframe = dataframe

        self.tokenizer = tokenizer
        self.mode = mode  # Options are 'pretrain' or 'finetune'

        self.mask_prob = mask_prob
        # self.max_length = max_length

    def corrupt_selfies(self, selfies_str):
        tokens = list(sf.split_selfies(selfies_str))
        corrupted = []
        for token in tokens:
            if random.random() < self.mask_prob:
                corrupted.append("[MASK]")
            else:
                corrupted.append(token)
        return "".join(corrupted)

    def __len__(self):
        """Return the total number of entries in the dataset."""
        lengths = self.dataframe.map_partitions(len).compute()
        total_len = lengths.sum()
        # return int(total_len)
        return total_len

    def __getitem__(self, idx):
        # print(f"self.tokenizer.mask_token_id: {self.tokenizer.mask_token_id}")
        selfies_string = str(self.dataframe.iloc[idx]["selfies"])

        # Corrupt only in 'pretrain' mode
        if self.mode == "pretrain":
            corrupted_selfies = self.corrupt_selfies(selfies_string)
        else:
            corrupted_selfies = selfies_string  # No corruption in finetune

        # Tokenize
        input_tokens = list(sf.split_selfies(corrupted_selfies))
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(input_tokens), dtype=torch.long
        )

        # Always use the original (uncorrupted) tokens as labels in pretrain
        label_tokens = list(sf.split_selfies(selfies_string))
        labels = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(label_tokens), dtype=torch.long
        )

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # Mask labels for unmasked tokens:
        mask_token_id = self.tokenizer.mask_token_id

        # for i, token_id in enumerate(input_ids):
        #     if token_id != mask_token_id:
        #         labels[i] = -100  # ignore loss on unmasked tokens

        # Suggested vectorized version:
        labels = torch.where(
            input_ids == mask_token_id, labels, torch.full_like(labels, -100)
        )

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return sample

    def encode_inhibition_site(self, inhibition_site):
        site = inhibition_site.strip().upper()
        return 0 if "RVP" in site else 1 if "RVE" in site else -1


class SelfiesFinetuneDataset(Dataset):
    def __init__(self, dataframe, tokenizer, mode="pretrain", mask_prob=0.15):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        self.dataframe = dataframe.reset_index(
            drop=True
        )  # Ensure indices are 0,1,2,...
        self.tokenizer = tokenizer
        self.mode = mode  # Options are 'pretrain' or 'finetune'

        self.mask_prob = mask_prob

    def corrupt_selfies(self, selfies_str):
        tokens = list(sf.split_selfies(selfies_str))
        corrupted = []
        for token in tokens:
            if random.random() < self.mask_prob:
                corrupted.append("[MASK]")
            else:
                corrupted.append(token)
        return "".join(corrupted)

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        selfies_string = str(self.dataframe.iloc[idx]["selfies"])

        # Corrupt only in 'pretrain' mode
        if self.mode == "pretrain":
            corrupted_selfies = self.corrupt_selfies(selfies_string)
        else:
            corrupted_selfies = selfies_string  # No corruption in finetune

        # Tokenize
        input_tokens = list(sf.split_selfies(corrupted_selfies))
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(input_tokens), dtype=torch.long
        )

        # Always use the original (uncorrupted) tokens as labels in pretrain
        label_tokens = list(sf.split_selfies(selfies_string))
        labels = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(label_tokens), dtype=torch.long
        )

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # Mask labels for unmasked tokens:
        mask_token_id = self.tokenizer.mask_token_id

        labels = torch.where(
            input_ids == mask_token_id, labels, torch.full_like(labels, -100)
        )

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return sample

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


class SelfiesIterableDataset(IterableDataset):
    def __init__(
        self,
        parquet_path,
        tokenizer,
        partitions=None,
        mode="pretrain",
        mask_prob=0.15,
        verbose=False,
    ):
        """
        Args:
            parquet_path (str): Path to the Parquet file(s).
            tokenizer: Tokenizer with convert_tokens_to_ids() and mask_token_id attributes.
            partitions (list[int], optional): Specific partition indices to load. If None, load all partitions.
            mode (str): 'pretrain' (corrupt input) or other modes (no corruption).
            mask_prob (float): Probability of masking a token during corruption.
            verbose (bool): Whether to print debug logs for each row.
        """
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.partitions = partitions
        self.mode = mode
        self.mask_prob = mask_prob
        self.verbose = verbose

        # Lazy-load for length computation
        self._df = dd.read_parquet(self.parquet_path)
        self._length = None

    def __len__(self):
        if self._length is None:
            if self.partitions is not None:
                partition_lengths = [
                    int(self._df.get_partition(i).map_partitions(len).compute().sum())
                    for i in self.partitions
                ]
                self._length = sum(partition_lengths)
            else:
                self._length = int(self._df.map_partitions(len).compute().sum())
        return self._length

    def corrupt_selfies(self, selfies_str):
        """Randomly mask tokens in a SELFIES string."""
        tokens = list(sf.split_selfies(selfies_str))
        corrupted = [
            "[MASK]" if random.random() < self.mask_prob else token for token in tokens
        ]
        return "".join(corrupted)

    def process_row(self, selfies_string):
        """Corrupt, tokenize, and create model input tensors."""
        corrupted_selfies = (
            self.corrupt_selfies(selfies_string)
            if self.mode == "pretrain"
            else selfies_string
        )

        # Tokenize input
        input_tokens = list(sf.split_selfies(corrupted_selfies))
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(input_tokens), dtype=torch.long
        )

        # Tokenize target labels
        label_tokens = list(sf.split_selfies(selfies_string))
        labels = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(label_tokens), dtype=torch.long
        )

        # Create attention mask
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        # Mask labels where input is not a mask token
        mask_token_id = self.tokenizer.mask_token_id
        labels = torch.where(
            input_ids == mask_token_id, labels, torch.full_like(labels, -100)
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self):
        """Iterate over all partitions and rows."""
        # df = dd.read_parquet(self.parquet_path)
        df = self._df
        partitions_to_iter = (
            self.partitions if self.partitions is not None else range(df.npartitions)
        )

        for partition_idx in partitions_to_iter:
            partition_df = df.get_partition(partition_idx).compute()

            for _, row in partition_df.iterrows():
                selfies_string = str(row["selfies"])
                if self.verbose:
                    print(f"[SelfiesDataset] Processing row: {selfies_string[:30]}...")

                yield self.process_row(selfies_string)


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
