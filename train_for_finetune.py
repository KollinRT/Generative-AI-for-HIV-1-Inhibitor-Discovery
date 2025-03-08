import argparse
import pandas as pd
import numpy as np
import yaml
from os.path import isfile
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies
from SelfiesDataHandler import SelfiesDataset, collate_fn_fine, NNLossHandler
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartConfig
from tokenizers import Tokenizer
import torch
from tqdm import tqdm  
import csv # FOR SAVING THE LOSS

def load_hyperparameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(args, key): # TODO: Integrate key into here.... where?
    try:
        df = pd.read_csv(args.selfies_dataset)
    except FileNotFoundError:
        from prepare_dataset import prepare_dataset_for_pretrain
        print("No SEFLIES dataset")
        prepare_dataset_for_pretrain(path=args.smiles_dataset, save_to=args.selfies_dataset)
        df = pd.read_csv(args.selfies_dataset)
    print("We have a SELFIES set for ya!")

    print("Creating SELFIES.txt for tokenization.")
    if not isfile(args.prepared_data_path):
        from prepare_dataset import create_selfies_file
        if args.subset_size != 0:
            create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path) # prepared_data_path is where the selfies by itself goes...
        else:
            create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path) # TODO: Need to ensure this is being done! LAST STEP!
    print("SELFIES .txt is ready for tokenization.")

    print("Creating file for training!")
    if not isfile(f"./data/trainable_selfies_{key}.csv"):
        from prepare_dataset import prepare_dataset_for_pretrain
        prepare_dataset_for_pretrain(f"./model_name_{key}.csv",f"./data/trainable_selfies_{key}.csv")
    print(f"File for training is ready! (trainable_selfies_{key}.csv)")
        

def finetune_BART(hyperparameters_dict, args, key):
    # TODO: Integrate this into the code!
    num_epochs = hyperparameters_dict[key]['TRAIN_EPOCHS']
    optimizer_selection = hyperparameters_dict[key]['optimizer']
    criterion_selection = hyperparameters_dict[key]['criterion']
    learning_rate = hyperparameters_dict[key]['LEARNING_RATE']

    # Define early stopping criteria
    early_stopping_toggle = hyperparameters_dict[key]["early_stopping_toggle"]
    early_stopping_threshold = hyperparameters_dict[key]["early_stopping_threshold"]
    early_stopping_patience = hyperparameters_dict[key]["early_stopping_patience"]

    loss_handler = NNLossHandler(
        loss_name=hyperparameters_dict[key]['criterion'],
        early_stopping_toggle=early_stopping_toggle,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience
    )

    df = pd.read_csv(f"./data/smiles_finetune_data_properties_selfies.csv")

    # DEBUG PURPOSES:
    # Set a random seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    # Split at 90%
    split_idx = int(len(df) * 0.9)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    # Create two separate DataFrames
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[test_indices].reset_index(drop=True)

    # Load the tokenizer and dataset
    # tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")
    # TODO: THIS IS WHERE THE FINE DATASET IS LOADED....
    # THIS IS THE SAME FINETUNING DATASET USED FOR ALL OF THE MODELS...
    # I CAN LOOP OVER THE KEYS BUT KEEP IT THE SAME BUT FEED IN THE DIFFERENT MODELS...
    # I ONLY NEED TO GET THE FINETUNED SELFIES DATASET ONCE... WITH GETDATAFINETUNE.... 
    # THIS CAN BE HARDCODED FOR BOTH!                                                 # make sure the tokenizer from pretrain is here also...
    # finetune_train_dataset = SelfiesDataset(csv_file=f"./data/smiles_finetune_data_properties_selfies.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='finetune')
    # finetune_validation_dataset = SelfiesDataset(csv_file=f"./data/smiles_finetune_data_properties_selfies.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='finetune')
    # Use the split DataFrames instead of the file path
    finetune_train_dataset = SelfiesDataset(dataframe=df_train, tokenizer_path=f"./data/bpe_filter_{key}/bpe.json",
                                            mode="finetune")
    finetune_validation_dataset = SelfiesDataset(dataframe=df_val, tokenizer_path=f"./data/bpe_filter_{key}/bpe.json",
                                                 mode="finetune")
    # TODO: NEW 02/18/2025 get the IC50 finetune data!

    # TODO: Above should be the one fine_tune approach. It is the same for all models...
    # TODO: NEW 02/18/2025... This should be a load the finetune train and validation models and randomly grab 90/10% split...
    #

    # Create DataLoader
    # finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_fine)
    # finetune_validation_loader = DataLoader(finetune_validation_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_fine)
    finetune_train_loader = DataLoader(
        finetune_train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_fine
    )
    finetune_validation_loader = DataLoader(
        finetune_validation_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_fine
    )

    # TODO: update with the format for the model it should be... the correct model name....
    # model = BartForConditionalGeneration.from_pretrained(f'./selfies_BART_pretrained_{key}.pth')
    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  # Load base BART model
    # state_dict = torch.load(f'./selfies_BART_pretrained_{key}.pth', map_location='cpu')  # Load state dict
    # model.load_state_dict(state_dict)  # Load weights into model
    model_path = f"./selfies_BART_pretrained_{key}/"
    model = BartForConditionalGeneration.from_pretrained(model_path)

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
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    if criterion_selection == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_selection == "nll":
        criterion = torch.nn.NLLLoss()
    elif criterion_selection == "poisson":
        criterion = torch.nn.PoissonNLLLoss()
    elif criterion_selection == "kldiv":
        criterion = torch.nn.KLDivLoss()
    elif criterion_selection == "bce":
        criterion = torch.nn.BCELoss()
    elif criterion_selection == "bcewithlogits":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif criterion_selection == "marginranking":
        criterion = torch.nn.MarginRankingLoss()
    elif criterion_selection == "hingeembedding":
        criterion = torch.nn.HingeEmbeddingLoss()
    elif criterion_selection == "multilabelsoftmargin":
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif criterion_selection == "smoothl1":
        criterion = torch.nn.SmoothL1Loss()
        
    # # Open a CSV file to save the epoch and loss
    # csv_file_path = f'./finetuning_loss_{key}.csv' # TODO: Update file names here!
    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(['Epoch', 'Loss'])  # Write the header
    #
    #
    #     # Check for CUDA
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model.to(device)
    #
    #     # Training loop
    #     model.train()      # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
    # #     for epoch in range(num_epochs):  # Number of epochs
    # #         total_loss = 0
    # #         for batch in tqdm(finetune_loader):
    # #             # set somewhere else... look into that...
    # #             input_ids = batch['input_ids'].to(device)
    # #             attention_mask = batch['attention_mask'].to(device)
    # #                                                             # makes sense I think?
    # #             # Forward pass (assuming self-supervised learning, labels = input_ids)
    # #             # TODO: Look into the model inputs.... it is just what is established above but should be good...
    # #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # #
    # #
    # #             # Compute loss and optimize
    # #             loss = outputs.loss
    # #             total_loss += loss.item()
    # #             optimizer.zero_grad()
    # #             loss.backward()
    # #             optimizer.step()
    # #         #TODO: DOES THIS LOOK RIGHT?
    # #         # if args.early_stopping == True:
    # #         #   if total_loss < args.early_stopping_threshold: # total_loss is going to get continuously larger?
    # #         #       break
    # #
    # #         # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(finetune_loader)}")
    # #         # Log the loss every 5 epochs
    # #         if (epoch + 1) % 5 == 0:
    # #             print(f"Epoch {epoch + 1}, Loss: {total_loss / len(finetune_loader)}")
    # #
    # #         # Save the epoch and loss to the CSV file
    # #         csv_writer.writerow([epoch + 1, total_loss / len(finetune_loader)])
    # #
    # #         # Early stopping (if enabled)
    # #         if args.early_stopping and total_loss < args.early_stopping_threshold:
    # #             print("Early stopping triggered")
    # #             break
    # #
    # #
    # # # Save the model                # TODO: this is replaced by the model key name...
    # # torch.save(model.state_dict(), f'./selfies_BART_finetuned_{key}.pth')
    # # # model.state_dict() or what else?
    # # for batch in tqdm(finetune_train_loader):
    # #     input_ids = batch['input_ids'].to(device)
    # #     attention_mask = batch['attention_mask'].to(device)
    # #
    # #     print(f"input_ids shape: {input_ids.shape}")
    # #     print(f"attention_mask shape: {attention_mask.shape}")
    # #
    # #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    #
    # for epoch in range(num_epochs):
    #     total_train_loss = 0
    #     for batch in tqdm(finetune_train_loader):
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    #         logits = outputs.logits
    #         target = input_ids
    #
    #         logits = logits.view(-1, logits.size(-1))
    #         target = target.view(-1)
    #
    #         loss = loss_handler.compute_loss(logits, target)
    #         total_train_loss += loss.item()
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     avg_train_loss = total_train_loss / len(finetune_train_loader)
    #
    #     total_val_loss = 0
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in tqdm(finetune_validation_loader):
    #             input_ids = batch['input_ids'].to(device)
    #             attention_mask = batch['attention_mask'].to(device)
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    #             logits = outputs.logits
    #             target = input_ids
    #
    #             logits = logits.view(-1, logits.size(-1))
    #             target = target.view(-1)
    #
    #             val_loss = loss_handler.compute_loss(logits, target)
    #             total_val_loss += val_loss.item()
    #
    #     avg_val_loss = total_val_loss / len(finetune_validation_loader)
    #
    #     print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
    #     csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
    #     csv_file.flush()
    #
    #     if early_stopping_toggle:
    #         if avg_val_loss < best_loss:
    #             best_loss = avg_val_loss
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #         if patience_counter >= early_stopping_patience:
    #             print(f"Early stopping triggered after epoch {epoch + 1}")
    #             break
    #
    # torch.save(model.state_dict(), f'./selfies_BART_finetuned_{key}.pth')
    # print(f"Model saved to ./selfies_BART_finetuned_{key}.pth")
    # Define the file path for saving the loss values
    csv_file_path = f'./finetuning_loss_{key}.csv'

    # ✅ Open the CSV file before training and keep it open
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])  # Write header

        # ✅ Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # ✅ Initialize best loss for early stopping
        best_loss = float('inf')
        patience_counter = 0

        # ✅ Training loop
        for epoch in range(num_epochs):
            total_train_loss = 0
            optimizer.zero_grad()

            # Training phase
            model.train()
            for batch in tqdm(finetune_train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                target = input_ids

                # Reshape for loss computation
                logits = logits.view(-1, logits.size(-1))
                target = target.view(-1)

                loss = loss_handler.compute_loss(logits, target)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(finetune_train_loader)

            # Validation phase
            total_val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(finetune_validation_loader, desc=f"Validation Epoch {epoch + 1}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    logits = outputs.logits
                    target = input_ids

                    # Reshape for loss computation
                    logits = logits.view(-1, logits.size(-1))
                    target = target.view(-1)

                    val_loss = loss_handler.compute_loss(logits, target)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(finetune_validation_loader)

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

            # ✅ Save the epoch loss to the CSV file
            csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
            csv_file.flush()

            # ✅ Early stopping logic
            if early_stopping_toggle:
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"🚨 Early stopping triggered after epoch {epoch + 1}")
                    break

    # ✅ Save the model after training completes
    torch.save(model.state_dict(), f'./selfies_BART_finetuned_{key}.pth')
    print(f"✅ Model saved to ./selfies_BART_finetuned_{key}.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
    parser.add_argument("--selfies_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
    parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
    parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters_path)
    print("Loaded hyperparameters:", hyperparameters)
    bart_hyperparameters = hyperparameters.get("BART", {}) # This would have some model specific hyperparameters and probably be part of a for loop to iterate over the keys representing each model in the hyperparameters dictionary.
    print("BART hyperparameters:", bart_hyperparameters)
    
    for key in bart_hyperparameters.keys():
        # Update paths based on the current key
        args.smiles_dataset=f"model_name_{key}.csv"
        print(args.smiles_dataset)
        args.selfies_dataset = f"./data/molecule_data_{key}.csv" # This is the prepared data path...
        # args.prepared_data_path = f"./data/{key}_prepared_data.txt"
        args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"
        args.bpe_path = f"./data/bpe_filter_{key}/"
                
        # Prepare data
        # prepare_data(args, key)

        # Train model
        finetune_BART(bart_hyperparameters, args, key)

if __name__ == "__main__":
    main()
    
"""
TODO: Finetuning data requires the specific dataset from the beginning...
it needs to have site_name for it... My pretrain data does NOT need it...
"""

# TODO: I need to add early_stopping to the hyperparameters...