import argparse
import pandas as pd
import yaml
from os.path import isfile
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies
from SelfiesDataHandler import SelfiesDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartConfig
from tokenizers import Tokenizer
import torch
from tqdm import tqdm  
import csv

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
    
    # TODO: Create the selfies.txt for the bpe tokenizer...
    # But to train the model we need to have the inhibition site and IC50 values and selfies
    # So training has the big dataset but bpe has little...
    
    # BPE is created for args.prepared_data_path... and saved to arg.bpe_path
    
    # Do I need to create two files? One for tokenizer and one for the selfies file with all columns
    # this would be the file spat out into the model...
    
    print("Creating BPE tokenizer.")
    if not isfile(args.bpe_path + "/merges.txt"):
        import prepare_dataset
        prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
    print("BPE Tokenizer is ready.")


def pretrain_BART(hyperparameters_dict, args, key):
    # TODO: Integrate this into the code!
    num_epochs = hyperparameters_dict[key]['TRAIN_EPOCHS']
    optimizer_selection = hyperparameters_dict[key]['optimizer']
    criterion_selection = hyperparameters_dict[key]['criterion']
    learning_rate = hyperparameters_dict[key]['LEARNING_RATE']
    
    # Load the tokenizer and dataset
    tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")
    pretrain_dataset = SelfiesDataset(csv_file=f"./data/trainable_selfies_{key}.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='pretrain')

    # Create DataLoader
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),  # Set vocab size including special tokens
        max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],  # Adjust based on your needs
        encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"], # TODO: make sure I can import this hyperparameters_dict correctly and interface it...
        decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
        encoder_attention_heads=hyperparameters_dict[key]["NUM_ENCODER_ATTENTION_HEADS"],
        decoder_attention_heads=hyperparameters_dict[key]["NUM_DECODER_ATTENTION_HEADS"],
        encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
        decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
        hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],  # Ensure this is divisible by the number of attention heads/ doesn't exist?
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=tokenizer.token_to_id("</s>"),
        mask_token_id=tokenizer.token_to_id("<mask>")  # Ensure this matches the ID used during pre-training # Not in OTHER!
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
    
    csv_file_path = f'./pretraining_loss_{key}.csv' # TODO: Update file names here!
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss'])  # Write the header

    
        # Check for CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training loop
        model.train()      # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
        for epoch in range(num_epochs):  # Number of epochs 
            total_loss = 0
            for batch in tqdm(pretrain_loader):
                # set somewhere else... look into that... 
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                                                                # makes sense I think?
                # Forward pass (assuming self-supervised learning, labels = input_ids)
                # TODO: Look into the model inputs.... it is just what is established above but should be good...
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                
                
                # Compute loss and optimize
                loss = outputs.loss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #TODO: DOES THIS LOOK RIGHT? 
            # if args.early_stopping == True:
            #   if total_loss < args.early_stopping_threshold: # total_loss is going to get continuously larger?
            #       break
            
            # Log the loss every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}, Loss: {total_loss / len(pretrain_loader)}")

            # Save the epoch and loss to the CSV file
            csv_writer.writerow([epoch + 1, total_loss / len(pretrain_loader)])

            # Early stopping (if enabled)
            if args.early_stopping and total_loss < args.early_stopping_threshold:
                print("Early stopping triggered")
                break

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(pretrain_loader)}")

        # Save the model                # TODO: this is replaced by the model key name...
        torch.save(model.state_dict(), f'./selfies_BART_pretrained_{key}.pth')
        # model.state_dict() or what else?


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
        prepare_data(args, key)

        # Train model
        pretrain_BART(bart_hyperparameters, args, key)

if __name__ == "__main__":
    main()
    

# NOTES BELOW TODO:
"""
I think I need to look at the model and make it a function that is configurable... 
I do not think that transformers API will work well for this...
sadly... Just get my old code training loop to work!

I have pretrain and finetune training code, I think?
I have the BartConfig also established in my WIP_BartSettings.py file...
    Look into that and see if I can get that to work...!
        Have hyperparameters also define the optimizer... and the loss function...
            It should be in here... I need to rewrite the workflow...

It is defined in `def train_and_save_BART(hyperparameters_dict, selfies_path="./data/selfies_subset.txt", bpe_path="./data/bpe/", save_to="./models/saved_model/"):`

NEED to differentiate between pretrain and fine-tune... Which has IC50/inhibition_site and which does not?

I think I have it in:
Main5Finetuning.py... work through refactoring the logic to work with this setup...
pretrain_dataset = SelfiesDataset(csv_file='./OrigFileSQL_Cleaned_SELFIES_READY.csv', tokenizer_path='./data/bpe/bpe.json', mode='finetune')
TODO: I think this involves pushing the current code, cleaning it up, then pushing the new restructured code?

TODO: New code is the old training routine... pytorch base training...
"""


"""
So, this code. It should check to see if the pretrained model exists, then if not train the model...
- - Or do I still want it do it for every one? Question is to filter at pre-train or at fine-tune? I could make a second file that has pre-train filtering and check those metrics... regardless, I need to get logging enabled along with early stopping?
"""
