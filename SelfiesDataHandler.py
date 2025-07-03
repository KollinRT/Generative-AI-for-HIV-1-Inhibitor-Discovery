import torch
import pandas as pd
import random
import selfies as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def collate_fn_pre(batch):
    # Pretraining mode: items only have 'input_ids'
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()  # Create attention mask (1 for tokens, 0 for padding)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def collate_fn_fine(batch):
    """Collate function for DataLoader to pad sequences dynamically."""

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    # Pad sequences to the max length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Updated to the right padding value
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)  # 0 for padding in mask

    batch_dict = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }

    # If finetune mode, include additional targets
    if 'IC50' in batch[0]:
        batch_dict['IC50'] = torch.stack([item['IC50'] for item in batch])
        batch_dict['inhibition_site'] = torch.stack([item['inhibition_site'] for item in batch])

    return batch_dict

def collate_fn(batch, mode='fine'):
    input_ids = [item['input_ids'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    ) if mode == 'fine' else (input_ids_padded != 0).long()

    batch_dict = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }

    if mode == 'fine':
        if 'IC50' in batch[0]:
            batch_dict['IC50'] = torch.stack([item['IC50'] for item in batch])
        if 'inhibition_site' in batch[0]:
            batch_dict['inhibition_site'] = torch.stack([item['inhibition_site'] for item in batch])

    return batch_dict


class SelfiesDataset(Dataset):
    # def __init__(self, csv_file, tokenizer_path, mode='pretrain'):
    def __init__(self, dataframe, tokenizer, mode='pretrain'):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        self.dataframe = dataframe.reset_index(drop=True)  # Ensure indices are 0,1,2,...


        self.tokenizer = tokenizer
        self.mode = mode  # Options are 'pretrain' or 'finetune'

        # self.mask_prob = mask_prob
        # self.max_length = max_length

    def corrupt_selfies(self, selfies_str):
        tokens = list(sf.split_selfies(selfies_str))
        corrupted = []
        for token in tokens:
            if random.random() < self.mask_prob:
                corrupted.append('[MASK]')
            else:
                corrupted.append(token)
        return ''.join(corrupted)

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        selfies_string = str(self.dataframe.iloc[idx]['selfies'])  # force cast to str

        # Correctly tokenize SELFIES using semantic splitting
        tokens = list(sf.split_selfies(selfies_string))  # ['[C]', '[C]', '[O]']
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

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
        return selfies_string.replace("<start>", "").replace("<end>", "").replace("<pad>", "")

class ClusteredSelfiesDataset(Dataset):
    def __init__(self, df, tokenizer, mode='pretrain'):
        """
        Initializes the dataset with a clustered DataFrame and a tokenizer.

        Args:
            df (pandas.DataFrame): DataFrame containing at least a 'selfies' column.
                                     (It may also contain additional columns like 'Cluster'.)
            tokenizer: A tokenizer instance with an .encode() method.
            mode (str): Either 'pretrain' or 'finetune'. In 'finetune' mode, additional columns
                        (e.g., 'IC50' and 'site_name') are expected.
        """
        # Reset index to ensure integer indexing (0, 1, 2, ...)
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use .iloc to ensure row-based access.
        row = self.df.iloc[idx]
        selfies_string = row['selfies']
        encoded = self.tokenizer.encode(selfies_string)

        if self.mode == 'pretrain':
            return {'input_ids': torch.tensor(encoded.ids, dtype=torch.long)}
        # no finetune mode in this pretrain only portion...
                # elif self.mode == 'finetune':
                #     # Expect additional columns for fine-tuning.
                #     IC50 = row.get('IC50', 0)  # default value if missing
                #     inhibition_site = row.get('site_name', "")
                #     inhibition_encoded = self.encode_inhibition_site(inhibition_site)
                #     return {
                #         'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
                #         'IC50': torch.tensor([IC50], dtype=torch.float),
                #         'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
                #     }
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

def mutate_selfies(selfies_string, mutation_rate=0.1):
    symbols = sf.split_selfies(selfies_string)  # Split the SELFIES into individual symbols
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

class NNLossHandler:
    def __init__(self, loss_name="crossentropy", early_stopping_toggle=False,
                 early_stopping_threshold=None, early_stopping_patience=5, **kwargs):
        """
        Initializes the loss handler with the specified loss function and early stopping configuration.

        Args:
            loss_name (str): Name of the loss function to use. Default is "crossentropy".
            early_stopping_toggle (bool): Whether to enable early stopping. Default is False.
            early_stopping_threshold (float): Loss threshold for early stopping. Default is None.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping. Default is 5.
            **kwargs: Additional arguments for initializing specific loss functions.
        """
        self.loss_name = loss_name.lower()
        self.loss_function = self._initialize_loss_function(self.loss_name, **kwargs)

        # Early stopping parameters
        self.early_stopping_toggle = early_stopping_toggle
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _initialize_loss_function(self, loss_name, **kwargs):
        """
        Maps the specified loss name to the corresponding PyTorch loss function.

        Args:
            loss_name (str): The name of the loss function.
            **kwargs: Additional arguments for the loss function.

        Returns:
            A PyTorch loss function instance.

        Raises:
            ValueError: If an invalid loss function name is provided.
        """
        if loss_name == "crossentropy":
            pad_token_id = kwargs.get("pad_token_id", 1)  # default to 1, can override
            return torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            # return torch.nn.CrossEntropyLoss()
        elif loss_name == "nll":
            return torch.nn.NLLLoss()
        elif loss_name == "poisson":
            return torch.nn.PoissonNLLLoss()
        elif loss_name == "kldiv":
            return torch.nn.KLDivLoss()
        elif loss_name == "bce":
            return torch.nn.BCELoss()
        elif loss_name == "bcewithlogits":
            return torch.nn.BCEWithLogitsLoss()
        elif loss_name == "marginranking":
            return torch.nn.MarginRankingLoss()
        elif loss_name == "hingeembedding":
            return torch.nn.HingeEmbeddingLoss()
        elif loss_name == "multilabelsoftmargin":
            return torch.nn.MultiLabelSoftMarginLoss()
        elif loss_name == "smoothl1":
            return torch.nn.SmoothL1Loss()
        # TODO NEW: Support more loss functions!
        else:
            raise ValueError(f"Invalid loss function name: {loss_name}")

    def get_loss_function(self):
        """
        Returns the initialized loss function.

        Returns:
            A PyTorch loss function instance.
        """
        return self.loss_function

    def compute_loss(self, outputs, targets):
        """
        Computes the loss given model outputs and targets.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        return self.loss_function(outputs, targets)

    def update_loss_function(self, loss_name, **kwargs):
        """
        Updates the loss function dynamically.

        Args:
            loss_name (str): Name of the new loss function.
            **kwargs: Additional arguments for initializing the new loss function.
        """
        self.loss_name = loss_name.lower()
        self.loss_function = self._initialize_loss_function(loss_name, **kwargs)

    def check_early_stopping(self, current_loss):
        """
        Checks whether early stopping should be triggered.

        Args:
            current_loss (float): The loss value of the current epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if not self.early_stopping_toggle:
            return False

        if self.early_stopping_threshold and current_loss < self.early_stopping_threshold:
            print(
                f"Early stopping triggered due to loss threshold: {current_loss:.4f} < {self.early_stopping_threshold:.4f}")
            return True

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0  # Reset patience counter
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} epochs without improvement.")
            return True

        return False
