# import torch
# import pandas as pd
# import random
# import selfies as sf
# from torch.utils.data import Dataset, DataLoader
# from transformers import BartForConditionalGeneration, BartConfig
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from tokenizers import Tokenizer
#
# def collate_fn(batch):
#     # If the dataset returns dictionaries (mode='finetune')
#     if isinstance(batch[0], dict):
#         # Padding input IDs
#         input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
#
#         # Stack IC50 and inhibition site data
#         ic50 = torch.stack([item['IC50'] for item in batch])
#         inhibition_site = torch.stack([item['inhibition_site'] for item in batch])
#
#         return {
#             'input_ids': input_ids,
#             'IC50': ic50,
#             'inhibition_site': inhibition_site
#         }
#     else:  # Pretraining mode, where only input_ids are expected
#         input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
#         return input_ids
#
#
# class SelfiesDataset(Dataset):
#     def __init__(self, csv_file, tokenizer_path, mode='pretrain'):
#         """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
#         self.mode = mode  # Options are 'pretrain' or 'finetune'
#
#     def __len__(self):
#         """Return the total number of entries in the dataset."""
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         """Retrieve an item by index."""
#         selfies_string = self.data.iloc[idx]['selfies'] # selfies not SELFIES
#         encoded = self.tokenizer.encode(selfies_string)
#
#         if self.mode == 'pretrain':
#             return torch.tensor(encoded.ids, dtype=torch.long)
#         elif self.mode == 'finetune':
#             IC50 = self.data.iloc[idx]['IC50']
#             inhibition_site = self.data.iloc[idx]['site_name']
#             inhibition_encoded = self.encode_inhibition_site(inhibition_site)
#
#             return {
#                 'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
#                 'IC50': torch.tensor([IC50], dtype=torch.float),
#                 'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
#             }
#
#     def encode_inhibition_site(self, inhibition_site):
#         """Encodes the inhibition site after normalizing string to prevent matching errors."""
#         inhibition_site = inhibition_site.strip().upper()  # Normalize by trimming spaces and converting to uppercase
#         if 'RVP' in inhibition_site:
#             return 0
#         elif 'RVE' in inhibition_site:
#             return 1
#         return -1  # Return -1 for cases where neither RVP nor RVE is found
#
# class SelfiesTokenizer:
#     def __init__(self, vocab):
#         self.vocab = vocab  # Assumes <start>, <end>, <pad>, <unk> are part of the vocab
#         self.inv_vocab = {v: k for k, v in vocab.items()}
#
#     def encode(self, selfies_string):
#         symbols = ["<start>"] + sf.split_selfies(selfies_string) + ["<end>"]
#         token_ids = [self.vocab.get(symbol, self.vocab["<unk>"]) for symbol in symbols]
#         return token_ids
#
#     def decode(self, token_ids):
#         symbols = [self.inv_vocab[id] for id in token_ids if id in self.inv_vocab]
#         selfies_string = "".join(symbols)
#         return selfies_string.replace("<start>", "").replace("<end>", "").replace("<pad>", "")
#
# def mutate_selfies(selfies_string, mutation_rate=0.1):
#     symbols = sf.split_selfies(selfies_string)  # Split the SELFIES into individual symbols
#     mutated_symbols = []
#     for symbol in symbols:
#         if random.random() < mutation_rate:
#             # Randomly replace a symbol with another
#             all_symbols = list(sf.get_semantic_robust_alphabet())
#             new_symbol = random.choice(all_symbols)
#             mutated_symbols.append(new_symbol)
#         else:
#             mutated_symbols.append(symbol)
#     mutated_selfies = "".join(mutated_symbols)  # Directly concatenate the symbols
#     return mutated_selfies
#
# def build_selfies_vocab(selfies_list):
#     unique_symbols = set()
#     # Ensure that selfies is a string before processing
#     for selfies in filter(lambda x: isinstance(x, str), selfies_list):
#         symbols = sf.split_selfies(selfies)
#         unique_symbols.update(symbols)
#     return {symbol: i for i, symbol in enumerate(unique_symbols)}
# from transformers import BartForConditionalGeneration, BartConfig
#
# # # Load the tokenizer to get the vocab size
# # tokenizer = Tokenizer.from_file("./data/bpe_non/bpe.json")
# #
# # config = BartConfig(
# #     vocab_size=tokenizer.get_vocab_size(),  # Set vocab size including special tokens
# #     max_position_embeddings=1024,  # Adjust based on your needs
# #     encoder_layers=6,
# #     decoder_layers=6,
# #     encoder_attention_heads=12,
# #     decoder_attention_heads=12,
# #     encoder_ffn_dim=3072,
# #     decoder_ffn_dim=3072,
# #     hidden_size=1152,  # Ensure this is divisible by the number of attention heads
# #     pad_token_id=tokenizer.token_to_id("<pad>"),
# #     bos_token_id=tokenizer.token_to_id("<s>"),
# #     eos_token_id=tokenizer.token_to_id("</s>"),
# #     mask_token_id=tokenizer.token_to_id("<mask>")  # Ensure this matches the ID used during pre-training
# # )
# #
# # model = BartForConditionalGeneration(config)
# #
# # # Load SELFIES data and build vocabulary
# # df = pd.read_csv("./ChEMBL34_druglike_activity.csv")
# # selfies_list = df['selfies'].tolist()
# # # vocab = build_selfies_vocab(selfies_list)
# # vocab = tokenizer.get_vocab()
# # inv_vocab = {v: k for k, v in vocab.items()}
# #
# # tokenizer = SelfiesTokenizer(vocab)
# #
# # pretrain_dataset = SelfiesDataset(csv_file='./ChEMBL34_druglike_activity.csv', tokenizer_path='./data/bpe_non/bpe.json', mode='finetune')
# # pretrain_loader = data_loader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# #
# # # Training setup
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # # TODO: One of the possible exploration areas. Different loss functions?
# # criterion = torch.nn.CrossEntropyLoss()
# #
# # # Check if CUDA (GPU support) is available, else use CPU
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using {device} device")
# #
# # model = torch.load('./selfies_BART_druglike_subset.pth')
# #
# #
# # # Move the model to the specified device
# # model = model.to(device)
# #
# # # TODO: Integrate tqdm GOT! Training loop is below!
# # # Training loop
# # num_epochs = 20  # Adjust based on your needs
# # model.train()
# # for epoch in range(num_epochs):
# #     for inputs in tqdm(pretrain_loader):  # Ensure correct DataLoader is referenced
# #         # Extracting input_ids and sending to device
# #         input_ids = inputs['input_ids'].to(device)
# #         # If you're also training with labels for fine-tuning, handle them similarly:
# #         # labels = inputs['labels_key'].to(device)  # Assuming a key for labels in your dataset
# #
# #         # Assuming a forward pass method that uses input_ids, and possibly labels
# #         outputs = model(input_ids=input_ids, labels=input_ids).logits
# #         # Calculate loss, assuming labels are your input_ids in this setup
# #         loss = criterion(outputs.transpose(1, 2), input_ids)  # Ensure loss calculation is correct based on the model output
# #
# #         # Optimizer steps
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #     print(f"Epoch {epoch}, Loss: {loss.item()}")
# #
# # # TODO: insert https://github.com/huggingface/accelerate and work on runpod once I get the samples sorted out to pretrain faster on a shit ton of data.
# # # Just need to get a big enough diverse corpus to utilize... Sample 100,000 or so, then train the pre-train using that.
# # # Then need to train the fine-tuning then figure out the model from there...
# #
# # # Saving the entire model
# # torch.save(model, './selfies_BART_druglike_fine_tuned.pth')


import torch
import pandas as pd
import random
import selfies as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

def collate_fn_pre(batch):
    # Pretraining mode: items only have 'input_ids'
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()  # Create attention mask (1 for tokens, 0 for padding)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def collate_fn_fine(batch):
    if isinstance(batch[0], dict):
        # Fine-tuning mode
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
        ic50 = torch.stack([item['IC50'] for item in batch])
        inhibition_site = torch.stack([item['inhibition_site'] for item in batch])

        return {
            'input_ids': input_ids,
            'IC50': ic50,
            'inhibition_site': inhibition_site
        }
    else:
        # Pretraining mode
        input_ids = pad_sequence(batch, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != 0).long(),  # Create an attention mask (1 for actual tokens, 0 for padding)
            # You may need to add more keys depending on what your model expects, like 'decoder_input_ids'
        }

class SelfiesDataset(Dataset):
    def __init__(self, csv_file, tokenizer_path, mode='pretrain'):
        """Initialize the dataset, loading data from CSV, setting up tokenizer and mode."""
        self.data = pd.read_csv(csv_file)
        # print(f"CSV columns: {self.data.columns.tolist()}")  # Debugging print statement
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.mode = mode  # Options are 'pretrain' or 'finetune'

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.data)

    # def __getitem__(self, idx):
    #     """Retrieve an item by index."""
    #     #print(f"Columns are: {self.data.columns}") # DEBUG
    #
    #     selfies_string = self.data.iloc[idx]['selfies']  # Update this if the column name is different
    #     encoded = self.tokenizer.encode(selfies_string)
    #
    #     if self.mode == 'pretrain':
    #         return torch.tensor(encoded.ids, dtype=torch.long)
    #     elif self.mode == 'finetune':
    #         IC50 = self.data.iloc[idx]['IC50']
    #         inhibition_site = self.data.iloc[idx]['site_name']
    #         inhibition_encoded = self.encode_inhibition_site(inhibition_site)
    #
    #         return {
    #             'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
    #             'IC50': torch.tensor([IC50], dtype=torch.float),
    #             'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
    #         }
    def __getitem__(self, idx):
        selfies_string = self.data.iloc[idx]['selfies']
        encoded = self.tokenizer.encode(selfies_string)
        if self.mode == 'pretrain':
            return {'input_ids': torch.tensor(encoded.ids, dtype=torch.long)}
        elif self.mode == 'finetune':
            IC50 = self.data.iloc[idx]['IC50']
            inhibition_site = self.data.iloc[idx]['site_name']
            inhibition_encoded = self.encode_inhibition_site(inhibition_site)
            return {
                'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
                'IC50': torch.tensor([IC50], dtype=torch.float),
                'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
            }

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
        elif self.mode == 'finetune':
            # Expect additional columns for fine-tuning.
            IC50 = row.get('IC50', 0)  # default value if missing
            inhibition_site = row.get('site_name', "")
            inhibition_encoded = self.encode_inhibition_site(inhibition_site)
            return {
                'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
                'IC50': torch.tensor([IC50], dtype=torch.float),
                'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
            }
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def encode_inhibition_site(self, inhibition_site):
        """
        Encodes the inhibition site string into a numerical label.

        Args:
            inhibition_site (str): The inhibition site string.

        Returns:
            int: 0 if the string contains 'RVP', 1 if it contains 'RVE', otherwise -1.
        """
        inhibition_site = inhibition_site.strip().upper()
        if 'RVP' in inhibition_site:
            return 0
        elif 'RVE' in inhibition_site:
            return 1
        return -1



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

# class NNLossHandler:
#     def __init__(self, loss_name="crossentropy", early_stopping_toggle=False,
#                  early_stopping_threshold=None, early_stopping_patience=5, **kwargs):
#         """
#         Initializes the loss handler with the specified loss function and early stopping configuration.
#
#         Args:
#             loss_name (str): Name of the loss function to use. Default is "crossentropy".
#             early_stopping_toggle (bool): Whether to enable early stopping. Default is False.
#             early_stopping_threshold (float): Loss threshold for early stopping. Default is None.
#             early_stopping_patience (int): Number of epochs to wait for improvement before stopping. Default is 5.
#             **kwargs: Additional arguments for initializing specific loss functions.
#         """
#         # self.loss_name = loss_name.lower()
#         # self.loss_function = self._initialize_loss_function(self.loss_name, **kwargs)
#
#         # Early stopping parameters
#         self.early_stopping_toggle = early_stopping_toggle
#         self.early_stopping_threshold = early_stopping_threshold
#         self.early_stopping_patience = early_stopping_patience
#         self.best_loss = float('inf')
#         self.patience_counter = 0

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
            return torch.nn.CrossEntropyLoss()
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

# """
# def pretrain_BART(hyperparameters_dict, args, key):
#     # TODO: Integrate this into the code!
#     num_epochs = hyperparameters_dict[key]['TRAIN_EPOCHS']
#     optimizer_selection = hyperparameters_dict[key]['optimizer']
#     criterion_selection = hyperparameters_dict[key]['criterion']
#     learning_rate = hyperparameters_dict[key]['LEARNING_RATE']
#     early_stopping_toggle = hyperparameters_dict[key]["early_stopping_toggle"]
#     early_stopping_threshold = hyperparameters_dict[key]["early_stopping_threshold"]
#     early_stopping_patience = hyperparameters_dict[key]["early_stopping_patience"]
#
#     # Load the tokenizer and dataset
#     tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")
#     pretrain_dataset = SelfiesDataset(csv_file=f"./data/trainable_selfies_{key}.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='pretrain')
#
#     # Create DataLoader
#     pretrain_loader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
#
#     config = BartConfig(
#         vocab_size=tokenizer.get_vocab_size(),  # Set vocab size including special tokens
#         max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],  # Adjust based on your needs
#         encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"], # TODO: make sure I can import this hyperparameters_dict correctly and interface it...
#         decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
#         encoder_attention_heads=hyperparameters_dict[key]["NUM_ENCODER_ATTENTION_HEADS"],
#         decoder_attention_heads=hyperparameters_dict[key]["NUM_DECODER_ATTENTION_HEADS"],
#         encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
#         decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
#         hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],  # Ensure this is divisible by the number of attention heads/ doesn't exist?
#         pad_token_id=tokenizer.token_to_id("<pad>"),
#         bos_token_id=tokenizer.token_to_id("<s>"),
#         eos_token_id=tokenizer.token_to_id("</s>"),
#         mask_token_id=tokenizer.token_to_id("<mask>")  # Ensure this matches the ID used during pre-training # Not in OTHER!
#     )
#
#
#     model = BartForConditionalGeneration(config)
#
#
#     if optimizer_selection == "adam":
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     elif optimizer_selection == "adamw":
#         optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     elif optimizer_selection == "sgd":
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     elif optimizer_selection == "adagrad":
#         optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
#     elif optimizer_selection == "adadelta":
#         optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
#     else:
#         raise ValueError(f"Invalid optimizer: {args.optimizer}")
#
#     if criterion_selection == "crossentropy":
#         criterion = torch.nn.CrossEntropyLoss()
#     elif criterion_selection == "nll":
#         criterion = torch.nn.NLLLoss()
#     elif criterion_selection == "poisson":
#         criterion = torch.nn.PoissonNLLLoss()
#     elif criterion_selection == "kldiv":
#         criterion = torch.nn.KLDivLoss()
#     elif criterion_selection == "bce":
#         criterion = torch.nn.BCELoss()
#     elif criterion_selection == "bcewithlogits":
#         criterion = torch.nn.BCEWithLogitsLoss()
#     elif criterion_selection == "marginranking":
#         criterion = torch.nn.MarginRankingLoss()
#     elif criterion_selection == "hingeembedding":
#         criterion = torch.nn.HingeEmbeddingLoss()
#     elif criterion_selection == "multilabelsoftmargin":
#         criterion = torch.nn.MultiLabelSoftMarginLoss()
#     elif criterion_selection == "smoothl1":
#         criterion = torch.nn.SmoothL1Loss()
#
#     csv_file_path = f'./pretraining_loss_{key}.csv' # TODO: Update file names here!
#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Epoch', 'Loss'])  # Write the header
#
#
#         # Check for CUDA
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         # Training loop variables initated for early stopping
#         best_loss = float('inf')
#         patience_counter = 0
#
#         # Training loop
#         model.train()      # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
#         for epoch in range(num_epochs):  # Number of epochs
#             total_loss = 0
#             for batch in tqdm(pretrain_loader):
#                 # set somewhere else... look into that...
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                                                                 # makes sense I think?
#                 # Forward pass (assuming self-supervised learning, labels = input_ids)
#                 # TODO: Look into the model inputs.... it is just what is established above but should be good...
#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#
#
#                 # Compute loss and optimize
#                 loss = outputs.loss
#                 total_loss += loss.item()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             #TODO: DOES THIS LOOK RIGHT?
#             # if args.early_stopping == True:
#             #   if total_loss < args.early_stopping_threshold: # total_loss is going to get continuously larger?
#             #       break
#
#             # Calculate average loss for the epoch
#             avg_loss = total_loss / len(pretrain_loader)
#
#             # Log the loss every 5 epochs
#             if (epoch + 1) % 5 == 0:
#                 print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
#
#             # Save the epoch and loss to the CSV file
#             csv_writer.writerow([epoch + 1, avg_loss])
#             csv_file.flush() # Ensure it writes at the end of each epoch
#
#             # Early stopping (if enabled)
#             if early_stopping_toggle:
#                 if avg_loss < best_loss:
#                     best_loss = avg_loss
#                     patience_counter = 0  # Reset patience counter if there's improvement
#                 else:
#                     patience_counter += 1
#
#                 if patience_counter >= early_stopping_patience:
#                     print(f"Early stopping triggered after epoch {epoch + 1}")
#                     break
#
#             print(f"Epoch {epoch + 1}, Loss: {avg_loss}") # This may be erased!
#
#         # Save the model                # TODO: this is replaced by the model key name...
#         torch.save(model.state_dict(), f'./selfies_BART_pretrained_{key}.pth')
#         # model.state_dict() or what else?
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
#     parser.add_argument("--selfies_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
#     parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
#     parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
#     args = parser.parse_args()
#
#     hyperparameters = load_hyperparameters(args.hyperparameters_path)
#     print("Loaded hyperparameters:", hyperparameters)
#     bart_hyperparameters = hyperparameters.get("BART", {}) # This would have some model specific hyperparameters and probably be part of a for loop to iterate over the keys representing each model in the hyperparameters dictionary.
#     print("BART hyperparameters:", bart_hyperparameters)
#
#     for key in bart_hyperparameters.keys():
#         # Update paths based on the current key
#         args.smiles_dataset=f"model_name_{key}.csv"
#         print(args.smiles_dataset)
#         args.selfies_dataset = f"./data/molecule_data_{key}.csv" # This is the prepared data path...
#         # args.prepared_data_path = f"./data/{key}_prepared_data.txt"
#         args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"
#         args.bpe_path = f"./data/bpe_filter_{key}/"
#
#         # Prepare data
#         prepare_data(args, key)
#
#         # Train model
#         pretrain_BART(bart_hyperparameters, args, key)
#
# if __name__ == "__main__":
#     main()
#
#
# # NOTES BELOW TODO:
# """
# I think I need to look at the model and make it a function that is configurable...
# I do not think that transformers API will work well for this...
# sadly... Just get my old code training loop to work!
#
# I have pretrain and finetune training code, I think?
# I have the BartConfig also established in my WIP_BartSettings.py file...
#     Look into that and see if I can get that to work...!
#         Have hyperparameters also define the optimizer... and the loss function...
#             It should be in here... I need to rewrite the workflow...
#
# It is defined in `def train_and_save_BART(hyperparameters_dict, selfies_path="./data/selfies_subset.txt", bpe_path="./data/bpe/", save_to="./models/saved_model/"):`
#
# NEED to differentiate between pretrain and fine-tune... Which has IC50/inhibition_site and which does not?
#
# I think I have it in:
# Main5Finetuning.py... work through refactoring the logic to work with this setup...
# pretrain_dataset = SelfiesDataset(csv_file='./OrigFileSQL_Cleaned_SELFIES_READY.csv', tokenizer_path='./data/bpe/bpe.json', mode='finetune')
# TODO: I think this involves pushing the current code, cleaning it up, then pushing the new restructured code?
#
# TODO: New code is the old training routine... pytorch base training...
# """
#
#
# """
# So, this code. It should check to see if the pretrained model exists, then if not train the model...
# - - Or do I still want it do it for every one? Question is to filter at pre-train or at fine-tune? I could make a second file that has pre-train filtering and check those metrics... regardless, I need to get logging enabled along with early stopping?
# """
#
#
# """
# #TODO: 12/20/2024: So, I need to write the code for finetuning... It needs to query the particular molecules? Or it needs to do them all?
# I think doing all the finetuning molecules rather than a subset is what is better!
# - This should mean finalizing the query for the Finetune molecules linked to the keys from the pretrained models.
#   this will allow for just continuation and running the model to adapt to just the 7k finetuned molecules of interest for HIV...
# """
#
# """


# # TODO: EARLY STOPPING BEST PRACTICES
# # Initialize variables for early stopping
# best_loss = float('inf')  # Lowest validation loss seen so far
# patience_counter = 0      # Counter for early stopping
# best_model_path = './best_model_weights.pth'  # File to save best model weights
#
# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     total_train_loss = 0
#
#     # Training phase
#     for batch in tqdm(pretrain_loader):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#
#         # Forward pass
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#
#         # Compute loss
#         logits = outputs.logits.view(-1, outputs.logits.size(-1))
#         target = input_ids.view(-1)
#         loss = loss_handler.compute_loss(logits, target)
#
#         total_train_loss += loss.item()
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     avg_train_loss = total_train_loss / len(pretrain_loader)
#
#     # Validation phase
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#             logits = outputs.logits.view(-1, outputs.logits.size(-1))
#             target = input_ids.view(-1)
#             val_loss = loss_handler.compute_loss(logits, target)
#             total_val_loss += val_loss.item()
#
#     avg_val_loss = total_val_loss / len(val_loader)
#     print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
#
#     # Early stopping and saving best weights
#     if avg_val_loss < best_loss:
#         best_loss = avg_val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), best_model_path)  # Save best weights
#         print(f"New best model saved with Validation Loss: {best_loss}")
#     else:
#         patience_counter += 1
#
#     if patience_counter >= early_stopping_patience:
#         print("Early stopping triggered!")
#         break

# LOAD THE BEST WEIGHTS
# Load the best model weights
# model.load_state_dict(torch.load(best_model_path))
# print("Best model weights loaded for testing.")

# Testing After Early Stopping (IF I AM USING TESTING?)
# model.eval()  # Set to evaluation mode
# total_test_loss = 0
# test_metrics = {}
#
# with torch.no_grad():
#     for batch in tqdm(test_loader):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#         logits = outputs.logits.view(-1, outputs.logits.size(-1))
#         target = input_ids.view(-1)
#         test_loss = loss_handler.compute_loss(logits, target)
#
#         total_test_loss += test_loss.item()
#
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss}")
#
