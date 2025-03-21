# import argparse
# import pandas as pd
# import yaml
# from os.path import isfile
# from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies
# from SelfiesDataHandler import SelfiesDataset, collate_fn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BartForConditionalGeneration, BartConfig
# from tokenizers import Tokenizer
# import torch
# from tqdm import tqdm
# import csv
# from SelfiesDataHandler import NNLossHandler
# import os
# from generateFingerprints import make_fingerprint_thisthat
# from generateClusters import cluster_molecules
#
# def load_hyperparameters(path):
#     with open(path, 'r') as file:
#         return yaml.safe_load(file)
#
# def prepare_data(args, key): # TODO: Integrate key into here.... where?
#     try:
#         df = pd.read_csv(args.selfies_dataset)
#     except FileNotFoundError:
#         from prepare_dataset import prepare_dataset_for_pretrain
#         print("No SEFLIES dataset")
#         prepare_dataset_for_pretrain(path=args.smiles_dataset, save_to=args.selfies_dataset)
#         df = pd.read_csv(args.selfies_dataset)
#     print("We have a SELFIES set for ya!")
#
#     print("Creating SELFIES.txt for tokenization.")
#     if not isfile(args.prepared_data_path):
#         from prepare_dataset import create_selfies_file
#         if args.subset_size != 0:
#             create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path) # prepared_data_path is where the selfies by itself goes...
#         else:
#             create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path) # TODO: Need to ensure this is being done! LAST STEP!
#     print("SELFIES .txt is ready for tokenization.")
#
#     print("Creating file for training!")
#     if not isfile(f"./data/trainable_selfies_{key}.csv"):
#         from prepare_dataset import prepare_dataset_for_pretrain
#         prepare_dataset_for_pretrain(f"./model_name_{key}.csv",f"./data/trainable_selfies_{key}.csv")
#     print(f"File for training is ready! (trainable_selfies_{key}.csv)")
#
#     # TODO: Create the selfies.txt for the bpe tokenizer...
#     # But to train the model we need to have the inhibition site and IC50 values and selfies
#     # So training has the big dataset but bpe has little...
#
#     # BPE is created for args.prepared_data_path... and saved to arg.bpe_path
#
#     # Do I need to create two files? One for tokenizer and one for the selfies file with all columns
#     # this would be the file spat out into the model...
#
#     print("Creating BPE tokenizer.")
#     if not isfile(args.bpe_path + "/merges.txt"):
#         import prepare_dataset
#         prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
#     print("BPE Tokenizer is ready.")
#
#
# def pretrain_BART(hyperparameters_dict, args, key):
#     # TODO: Integrate this into the code!
#
#     # Define the learning hyperparameters
#     num_epochs = hyperparameters_dict[key]['TRAIN_EPOCHS']
#     learning_rate = hyperparameters_dict[key]['LEARNING_RATE']
#     optimizer_selection = hyperparameters_dict[key]['optimizer']
#     # criterion_selection = hyperparameters_dict[key]['criterion']
#
#     # Define early stopping criteria
#     early_stopping_toggle = hyperparameters_dict[key]["early_stopping_toggle"]
#     early_stopping_threshold = hyperparameters_dict[key]["early_stopping_threshold"]
#     early_stopping_patience = hyperparameters_dict[key]["early_stopping_patience"]
#
#     loss_handler = NNLossHandler(
#         loss_name=hyperparameters_dict[key]['criterion'],
#         early_stopping_toggle=early_stopping_toggle,
#         early_stopping_threshold=early_stopping_threshold,
#         early_stopping_patience=early_stopping_patience
#     )
#
#
#     # Load the tokenizer and dataset
#     tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")
#     pretrain_dataset = SelfiesDataset(csv_file=f"./data/trainable_selfies_{key}.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='pretrain')
#
#     # TODO NEW: Add training and validation set...
#     # TODO NEW: Make it only need to run pretrain once for the whole set. So check if the model file exists and if so throw an error and say model already exists!
#     # This is done in the main portion!
#     # pretrain_train_dataset = SelfiesDataset(csv_file=f"./data/trainable_selfies_{key}.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='pretrain')
#     # pretrain_val_dataset = SelfiesDataset(csv_file=f"./data/trainable_selfies_{key}.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='pretrain')
#
#     # Create DataLoader
#     # TODO NEW: Get batch_size from the combined_config.yml
#     # pretrain_loader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
#     # pretrain_val_loader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
#
#     # TODO: NEW: If I already have the pretrain train-val split, do I have to redo it?
#     # I think just load it in and take the 90% length and do train_dataset[0:90%] val_dataset[90%+1:] then I can just do
#     # it like that? But I have a SelfiesDataset class that has mode = 'pretrain', I think that I can just incorporate the
#     # above logic into that? How would it be split though into two sets?
#     """
#     This would be something such as
#     df = pd.read_csv(f"/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/molecule_with_clusters_lsh_256perm.csv")
#     df = df.sort_values(by=['Cluster'])
#     mol_count = int(len(df)*0.9)
#     # print(int(len(df)/10))
#
#     # Then we need to sort it by clusterID and grab the lowest...
#
#     train = df[:mol_count] # 90%
#     valid = df[mol_count:] # 10%
#     pretrain_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate_fn)
#     valid_loader = DataLoader(valid, batch_size=16, shuffle=True, collate_fn=collate_fn)
#
#     This should work for both parts of the training loop I think! Just have the two loops for training and validation.
#     """
#
#     df = pd.read_csv(
#         f"./data/trainable_selfies_{key}.csv")
#     df = make_fingerprint_thisthat(df)
#     df = cluster_molecules(df, f"./data/trainable_selfies_{key}.csv")
#     print(df.columns)
#     print(df.head(2))
#     df = df.sort_values(by=['Cluster']) # TODO: NEW need to add clustering here... So this needs to be down with the fingerprinting work...
#     mol_count = int(len(df) * 0.9) # Get the fingerprint stuffs working here.
#     # print(int(len(df)/10))
#
#
#     # Then we need to sort it by clusterID and grab the lowest...
#
#     train = df[:mol_count]  # 90% train
#     valid = df[mol_count:]  # 10% val
#     pretrain_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(valid, batch_size=16, shuffle=True, collate_fn=collate_fn)
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
#     # if criterion_selection == "crossentropy":
#     #     criterion = torch.nn.CrossEntropyLoss()
#     # elif criterion_selection == "nll":
#     #     criterion = torch.nn.NLLLoss()
#     # elif criterion_selection == "poisson":
#     #     criterion = torch.nn.PoissonNLLLoss()
#     # elif criterion_selection == "kldiv":
#     #     criterion = torch.nn.KLDivLoss()
#     # elif criterion_selection == "bce":
#     #     criterion = torch.nn.BCELoss()
#     # elif criterion_selection == "bcewithlogits":
#     #     criterion = torch.nn.BCEWithLogitsLoss()
#     # elif criterion_selection == "marginranking":
#     #     criterion = torch.nn.MarginRankingLoss()
#     # elif criterion_selection == "hingeembedding":
#     #     criterion = torch.nn.HingeEmbeddingLoss()
#     # elif criterion_selection == "multilabelsoftmargin":
#     #     criterion = torch.nn.MultiLabelSoftMarginLoss()
#     # elif criterion_selection == "smoothl1":
#     #     criterion = torch.nn.SmoothL1Loss()
#
#     # csv_file_path = f'./pretraining_loss_{key}.csv' # TODO: Update file names here!
#     # with open(csv_file_path, mode='w', newline='') as csv_file:
#     #     csv_writer = csv.writer(csv_file)
#     #     csv_writer.writerow(['Epoch', 'Loss'])  # Write the header
#     #
#     #
#     #     # Check for CUDA
#     #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #     model.to(device)
#     #
#     #     # Training loop variables initated for early stopping
#     #     best_loss = float('inf')
#     #     patience_counter = 0
#     #
#     #     # Training loop
#     #     model.train()      # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
#     #     for epoch in range(num_epochs):  # Number of epochs
#     #         total_loss = 0
#     #         for batch in tqdm(pretrain_loader): # TODO: Need to do for training and for validation... implement scaffold splitting!q
#     #             # set somewhere else... look into that...
#     #             input_ids = batch['input_ids'].to(device)
#     #             attention_mask = batch['attention_mask'].to(device)
#     #                                                             # makes sense I think?
#     #             # # Forward pass (assuming self-supervised learning, labels = input_ids)
#     #             # # TODO: Look into the model inputs.... it is just what is established above but should be good...
#     #             # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#     #             #
#     #             #
#     #             # # Compute loss and optimize
#     #             # loss = outputs.loss
#     #             # total_loss += loss.item()
#     #             # optimizer.zero_grad()
#     #             # loss.backward()
#     #             # optimizer.step()
#     #
#     #             # Forward pass
#     #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#     #
#     #             # # Compute loss using the loss handler
#     #             # loss = loss_handler.compute_loss(outputs.loss, input_ids)
#     #
#     #             # Extract logits and reshape for CrossEntropyLoss
#     #             # https://huggingface.co/transformers/v4.4.2/model_doc/bart.html
#     #             logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
#     #             # print(logits)
#     #             # print(logits.shape)
#     #             target = input_ids  # Shape: (batch_size, sequence_length)
#     #
#     #             # Flatten logits and targets
#     #             logits = logits.view(-1, logits.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
#     #             target = target.view(-1)  # Shape: (batch_size * sequence_length)
#     #
#     #             # Compute loss
#     #             loss = loss_handler.compute_loss(logits, target) # Single scalar return value...
#     #             # print(loss)
#     #             # print(loss.shape)
#     #
#     #             total_loss += loss.item()
#     #
#     #             # Backward pass and optimization
#     #             optimizer.zero_grad()
#     #             loss.backward()
#     #             optimizer.step()
#     #
#     #         # TODO NEW: Need to add the evaluation of validation set! Implement scaffold training and stuffs!
#     #
#     #         #TODO: DOES THIS LOOK RIGHT?
#     #         # if args.early_stopping == True:
#     #         #   if total_loss < args.early_stopping_threshold: # total_loss is going to get continuously larger?
#     #         #       break
#     #
#     #         # Calculate average loss for the epoch
#     #         avg_loss = total_loss / len(pretrain_loader)
#     #
#     #         # Log the loss every 5 epochs
#     #         if (epoch + 1) % 5 == 0:
#     #             print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
#     #
#     #         # Save the epoch and loss to the CSV file
#     #         csv_writer.writerow([epoch + 1, avg_loss])
#     #         csv_file.flush() # Ensure it writes at the end of each epoch
#     #
#     #         # Early stopping (if enabled)
#     #         if early_stopping_toggle:
#     #             if avg_loss < best_loss:
#     #                 best_loss = avg_loss
#     #                 patience_counter = 0  # Reset patience counter if there's improvement
#     #             else:
#     #                 patience_counter += 1
#     #
#     #             if patience_counter >= early_stopping_patience:
#     #                 print(f"Early stopping triggered after epoch {epoch + 1}")
#     #                 break
#     #
#     #         print(f"Epoch {epoch + 1}, Loss: {avg_loss}") # This may be erased!
#     #
#     #         # Validation loop
#     #         model.eval()
#     #         val_metrics = evaluate_model_on_validation(model, val_set)
#     #         print(f"Epoch {epoch + 1}, Validation Metrics: {val_metrics}")
#     csv_file_path = f'./pretraining_loss_{key}.csv'  # TODO: Update file names here!
#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])  # Write the header
#
#         # Check for CUDA
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         # Training loop variables initiated for early stopping
#         best_loss = float('inf')
#         patience_counter = 0
#
#         # Training loop
#         model.train()  # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
#         for epoch in range(num_epochs):  # Number of epochs
#             total_train_loss = 0
#
#             # Training loop
#             model.train()
#             for batch in tqdm(pretrain_loader):  # Training data
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#
#                 # Forward pass
#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#
#                 # Extract logits and reshape for CrossEntropyLoss
#                 logits = outputs.logits
#                 target = input_ids
#
#                 # Flatten logits and targets
#                 logits = logits.view(-1, logits.size(-1))
#                 target = target.view(-1)
#
#                 # Compute loss
#                 loss = loss_handler.compute_loss(logits, target)
#
#                 total_train_loss += loss.item()
#
#                 # Backward pass and optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             # Calculate average training loss for the epoch
#             avg_train_loss = total_train_loss / len(pretrain_loader)
#
#             # Validation loop
#             total_val_loss = 0
#             model.eval()  # Set model to evaluation mode
#             with torch.no_grad():
#                 for batch in tqdm(val_loader):  # Validation data
#                     input_ids = batch['input_ids'].to(device)
#                     attention_mask = batch['attention_mask'].to(device)
#
#                     # Forward pass
#                     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#
#                     # Extract logits and reshape for CrossEntropyLoss
#                     logits = outputs.logits
#                     target = input_ids
#
#                     # Flatten logits and targets
#                     logits = logits.view(-1, logits.size(-1))
#                     target = target.view(-1)
#
#                     # Compute loss
#                     val_loss = loss_handler.compute_loss(logits, target)
#
#                     total_val_loss += val_loss.item()
#
#             # Calculate average validation loss for the epoch
#             avg_val_loss = total_val_loss / len(val_loader)
#
#             # TODO: Add PPL perplexity
#
#             # Log the losses
#             print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
#
#             # Save the epoch and losses to the CSV file
#             csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
#             csv_file.flush()  # Ensure it writes at the end of each epoch
#
#             # Early stopping (if enabled)
#             if early_stopping_toggle:
#                 if avg_val_loss < best_loss:
#                     best_loss = avg_val_loss
#                     patience_counter = 0  # Reset patience counter if there's improvement
#                 else:
#                     patience_counter += 1
#
#                 if patience_counter >= early_stopping_patience:
#                     print(f"Early stopping triggered after epoch {epoch + 1}")
#                     break
#
#         # Save the model                # TODO: this is replaced by the model key name...
#         torch.save(model.state_dict(), f'./selfies_BART_pretrained_{key}.pth') # TODO: This is the trained model name... Check to see if this exists at the start of the function and if so print that then skip!
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
#         if os.path.exists(f'./selfies_BART_pretrained_{key}.pth'): # TODO NEW: Maybe change this to a single file name. If I am not passing in a key name, why care for this? Only need to pretrain one model! ALL!
#         # TODO NEW: Does this get passed into someone in the train_for_finetune that we use all the same only file? I think this is done already!
#             print("Model already exists! No need to retrain")
#         else:
#             # Update paths based on the current key
#             args.smiles_dataset=f"model_name_{key}.csv"
#             print(args.smiles_dataset)
#             args.selfies_dataset = f"./data/molecule_data_{key}.csv" # This is the prepared data path...
#             # args.prepared_data_path = f"./data/{key}_prepared_data.txt"
#             args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"
#             args.bpe_path = f"./data/bpe_filter_{key}/"
#
#             # Prepare data
#             prepare_data(args, key)
#
#             # Train model
#             pretrain_BART(bart_hyperparameters, args, key)
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

import argparse
import pandas as pd
import yaml
from os.path import isfile
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies
from SelfiesDataHandler import SelfiesDataset, collate_fn_pre, NNLossHandler
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartConfig
from tokenizers import Tokenizer
import torch
from tqdm import tqdm
import csv
import os
from generateFingerprints import make_fingerprint_thisthat
from generateClusters import cluster_molecules


def load_hyperparameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def prepare_data(args, key):
    try:
        df = pd.read_csv(args.selfies_dataset)
    except FileNotFoundError:
        from prepare_dataset import prepare_dataset_for_pretrain
        print("No SELFIES dataset")
        prepare_dataset_for_pretrain(path=args.smiles_dataset, save_to=args.selfies_dataset)
        df = pd.read_csv(args.selfies_dataset)
    print("We have a SELFIES set for ya!")

    print("Creating SELFIES.txt for tokenization.")
    if not isfile(args.prepared_data_path):
        from prepare_dataset import create_selfies_file
        if args.subset_size != 0:
            create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
        else:
            create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
    print("SELFIES .txt is ready for tokenization.")

    print("Creating file for training!")
    if not isfile(f"./data/trainable_selfies_{key}.csv"):
        from prepare_dataset import prepare_dataset_for_pretrain
        prepare_dataset_for_pretrain(f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv")
    print(f"File for training is ready! (trainable_selfies_{key}.csv)")

    print("Creating BPE tokenizer.")
    if not isfile(args.bpe_path + "/merges.txt"):
        import prepare_dataset
        prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
    print("BPE Tokenizer is ready.")


# === New: ClusteredSelfiesDataset ===
class ClusteredSelfiesDataset(Dataset):
    def __init__(self, df, tokenizer, mode='pretrain'):
        """
        Args:
            df (pandas.DataFrame): Clustered DataFrame that must contain a 'selfies' column.
            tokenizer: A tokenizer instance with an .encode() method.
            mode (str): 'pretrain' or 'finetune'. In 'finetune' mode, additional columns (e.g., 'IC50', 'site_name') are expected.
        """
        self.df = df.reset_index(drop=True)  # Ensure indices are 0,1,2,...
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use .iloc to get the row by integer index.
        row = self.df.iloc[idx]
        selfies_string = row['selfies']
        encoded = self.tokenizer.encode(selfies_string)
        if self.mode == 'pretrain':
            return {'input_ids': torch.tensor(encoded.ids, dtype=torch.long)}
        elif self.mode == 'finetune':
            IC50 = row['IC50']
            inhibition_site = row['site_name']
            inhibition_encoded = self.encode_inhibition_site(inhibition_site)
            return {
                'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
                'IC50': torch.tensor([IC50], dtype=torch.float),
                'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
            }
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def encode_inhibition_site(self, inhibition_site):
        inhibition_site = inhibition_site.strip().upper()
        if 'RVP' in inhibition_site:
            return 0
        elif 'RVE' in inhibition_site:
            return 1
        return -1


# ===================================

def pretrain_BART(hyperparameters_dict, args, key):
    # Define learning hyperparameters
    num_epochs = hyperparameters_dict[key]['TRAIN_EPOCHS']
    learning_rate = hyperparameters_dict[key]['LEARNING_RATE']
    optimizer_selection = hyperparameters_dict[key]['optimizer']

    # Define early stopping criteria
    early_stopping_toggle = hyperparameters_dict[key]["early_stopping_toggle"]
    early_stopping_threshold = hyperparameters_dict[key]["early_stopping_threshold"]
    early_stopping_patience = hyperparameters_dict[key]["early_stopping_patience"]

    # TODO: NEW 03/11/2025: Work to utilize a LRScheduler (https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/)
    # torch.optim.lr_scheduler.ReduceLROnPlateau

    # Define learning rate scheduler
    learning_rate_scheduler_selection = hyperparameters_dict[key]['lr_sched']

    loss_handler = NNLossHandler(
        loss_name=hyperparameters_dict[key]['criterion'],
        early_stopping_toggle=early_stopping_toggle,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience
    )

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")

    # Load and process the DataFrame: apply fingerprinting and clustering
    df = pd.read_csv(f"./data/trainable_selfies_{key}.csv")
    # df = make_fingerprint_thisthat(df)
    # df = cluster_molecules(df, f"./data/trainable_selfies_{key}.csv")
    print(df.columns)
    print(df.head(2))

    # Sort by cluster and split into training and validation sets
    df = df.sort_values(by=['Cluster'])
    mol_count = int(len(df) * 0.9)
    train_df = df[:mol_count]  # 90% for training
    valid_df = df[mol_count:]  # 10% for validation

    # Instead of passing the raw DataFrame to DataLoader, wrap it in the new ClusteredSelfiesDataset
    train_dataset = ClusteredSelfiesDataset(train_df, tokenizer, mode='pretrain')
    valid_dataset = ClusteredSelfiesDataset(valid_df, tokenizer, mode='pretrain')

    pretrain_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_pre)
    val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_pre)

    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],
        encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"],
        decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
        encoder_attention_heads=hyperparameters_dict[key]["NUM_ENCODER_ATTENTION_HEADS"],
        decoder_attention_heads=hyperparameters_dict[key]["NUM_DECODER_ATTENTION_HEADS"],
        encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
        decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
        hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=tokenizer.token_to_id("</s>"),
        mask_token_id=tokenizer.token_to_id("<mask>")
    )

    model = BartForConditionalGeneration(config)

    if optimizer_selection == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_selection == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_selection == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_selection == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_selection == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_selection}")


    # TODO: NEW 03/11/2025: Work to utilize a LRScheduler (https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/)
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    if learning_rate_scheduler_selection == "ReduceLROnPlateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif learning_rate_scheduler_selection == "LinearLR":
        lr_sched = torch.optim.lr_scheduler.LinearLR(optimizer)
    elif learning_rate_scheduler_selection is None:
        lr_sched = None
    else:
        raise ValueError(f"Invalid optimizer: {learning_rate_scheduler_selection}")

    """
    learning_rate_scheduler_selection = hyperparameters_dict[key]['lr_sched']

    if learning_rate_scheduler_selection == "ReduceLROnPlateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif learning_rate_scheduler_selection == "LinearLR":
        lr_sched = torch.optim.lr_scheduler.LinearLR(optimizer)  # Fixed parentheses
    elif learning_rate_scheduler_selection == "":
        lr_sched = None  # Do not use any scheduler
    else:
        raise ValueError(f"Invalid optimizer: {learning_rate_scheduler_selection}")
        
    # Apply the learning rate scheduler only if it's set IN THE TRAINING LOOP!
    if lr_sched is not None:
        lr_sched.step()
    """


    # Print DataFrame info for debugging
    print("Final training DataFrame:")
    print(train_df.head(2))

    # Training loop
    csv_file_path = f'./pretraining_loss_{key}.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'learning_rate'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        best_loss = float('inf')
        patience_counter = 0

        model.train()
        for epoch in range(num_epochs):
            total_train_loss = 0
            for batch in tqdm(pretrain_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                target = input_ids

                logits = logits.view(-1, logits.size(-1))
                target = target.view(-1)

                loss = loss_handler.compute_loss(logits, target)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(pretrain_loader)

            total_val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    logits = outputs.logits
                    target = input_ids

                    logits = logits.view(-1, logits.size(-1))
                    target = target.view(-1)

                    val_loss = loss_handler.compute_loss(logits, target)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            # Adjust Learning Rate
            if lr_sched is not None:
                lr_sched.step()

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
            if lr_sched is not None:
                csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, lr_sched.get_last_lr()])
            else:
                csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, "default"])

            csv_file.flush()

            if early_stopping_toggle:
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break

        # torch.save(model.state_dict(), f'./selfies_BART_pretrained_{key}.pth')
        model.save_pretrained(f'./selfies_BART_pretrained_{key}')
        print(f"Model saved to ./selfies_BART_pretrained_{key}.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv",
                        help="Path of the SMILES dataset.")
    parser.add_argument("--selfies_dataset", required=False, metavar="/path/to/dataset/*.csv",
                        help="Path of the SELFIES dataset.")
    parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0,
                        help="Subset size to use (0 for full dataset).")
    parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/",
                        help="Path to hyperparameters YAML file.")
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters_path)
    print("Loaded hyperparameters:", hyperparameters) # TODO: NEW 02/16/2025 figure out why BART is empty in combined_config... I THINK IT WORKS... ✓✓
    bart_hyperparameters = hyperparameters.get("BART", {})
    print("BART hyperparameters:", bart_hyperparameters)

    for key in bart_hyperparameters.keys():
        if os.path.exists(f'./selfies_BART_pretrained_{key}.pth'):
            print("Model already exists! No need to retrain")
        else:
            args.smiles_dataset = f"model_name_{key}.csv"
            print(args.smiles_dataset)
            args.selfies_dataset = f"./data/molecule_data_{key}.csv"
            args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"
            args.bpe_path = f"./data/bpe_filter_{key}/"

            prepare_data(args, key)
            pretrain_BART(bart_hyperparameters, args, key)


if __name__ == "__main__":
    main()
