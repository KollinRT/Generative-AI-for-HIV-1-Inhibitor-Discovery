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

# def collate_fn(batch):
#     # If the dataset returns dictionaries (mode='finetune')
#     if isinstance(batch[0], dict):
#         # Padding input IDs
#         input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)

#         # Stack IC50 and inhibition site data
#         ic50 = torch.stack([item['IC50'] for item in batch])
#         inhibition_site = torch.stack([item['inhibition_site'] for item in batch])

#         return {
#             'input_ids': input_ids,
#             'IC50': ic50,
#             'inhibition_site': inhibition_site
#         }
#     else:  # Pretraining mode, where only input_ids are expected
#         input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
#         return input_ids

def collate_fn(batch):
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
        print(f"CSV columns: {self.data.columns.tolist()}")  # Debugging print statement
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.mode = mode  # Options are 'pretrain' or 'finetune'

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        selfies_string = self.data.iloc[idx]['selfies']  # Update this if the column name is different
        encoded = self.tokenizer.encode(selfies_string)

        if self.mode == 'pretrain':
            return torch.tensor(encoded.ids, dtype=torch.long)
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
