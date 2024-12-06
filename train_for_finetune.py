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
    
    # Load the tokenizer and dataset
    tokenizer = Tokenizer.from_file(f"./data/bpe_filter_{key}/bpe.json")
    # TODO: THIS IS WHERE THE FINE DATASET IS LOADED....
    # THIS IS THE SAME FINETUNING DATASET USED FOR ALL OF THE MODELS...
    # I CAN LOOP OVER THE KEYS BUT KEEP IT THE SAME BUT FEED IN THE DIFFERENT MODELS...
    # I ONLY NEED TO GET THE FINETUNED SELFIES DATASET ONCE... WITH GETDATAFINETUNE.... 
    # THIS CAN BE HARDCODED FOR BOTH!                                                 # make sure the tokenizer from pretrain is here also...
    finetune_dataset = SelfiesDataset(csv_file=f"./data/smiles_finetune_data_properties.csv", tokenizer_path=f"./data/bpe_filter_{key}/bpe.json", mode='finetune')

    # Create DataLoader
    finetune_loader = DataLoader(finetune_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # TODO: update with the format for the model it should be... the correct model name....
    model = BartForConditionalGeneration.from_pretrained(f'./selfies_BART_pretrained_{key}.pth')    
    
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
        
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()      # set epoch via hyperparameters... config... THINK ABOUT EARLY STOPPING ALSO!
    for epoch in range(num_epochs):  # Number of epochs 
        total_loss = 0
        for batch in tqdm(finetune_loader):
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
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(finetune_loader)}")

    # Save the model                # TODO: this is replaced by the model key name...
    torch.save(model.state_dict(), f'./selfies_BART_finetuned_{key}.pth')
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
        finetune_BART(bart_hyperparameters, args, key)

if __name__ == "__main__":
    main()
    
"""
TODO: Finetuning data requires the specific dataset from the beginning...
it needs to have site_name for it... My pretrain data does NOT need it...
"""