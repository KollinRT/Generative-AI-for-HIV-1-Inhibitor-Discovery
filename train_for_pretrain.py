# # # import argparse
# # #
# # # parser = argparse.ArgumentParser()
# # # parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
# # # parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
# # # parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
# # # parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
# # # parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
# # # # parser.add_argument("--roberta_fast_tokenizer_path", required=True, metavar="/path/to/robertafasttokenizer/", help="Directory of the RobertaTokenizerFast tokenizer. RobertaFastTokenizer only depends on the BPE Tokenizer and will be created regardless of whether it exists or not.")
# # # parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
# # # args = parser.parse_args()
# # #
# # # import pandas as pd
# # #
# # # # TODO: NEED TO apply the selfies conversion to build it... this would use the prepared_data_path, if I am looking right?
# # #
# # # try:
# # #     df = pd.read_csv(args.selfies_dataset)
# # # except FileNotFoundError:
# # #     from prepare_dataset import prepare_dataset_for_pretrain
# # #     print("No SEFLIES dataset")
# # #     prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
# # #     df = pd.read_csv(args.selfies_dataset)
# # # print("We have a SELFIES set for ya!")
# # #
# # # print("Creating SELFIES.txt for tokenization.")
# # # from os.path import isfile  # returns True if the file exists else False.
# # #
# # # if not isfile(args.prepared_data_path):
# # #     from prepare_dataset import create_selfies_file
# # #
# # #     if args.subset_size != 0:
# # #         create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
# # #     else:
# # #         create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
# # # print("SELFIES .txt is ready for tokenization.")
# # #
# # # print("Creating BPE tokenizer.")
# # # if not isfile(args.bpe_path + "/merges.txt"):
# # #     import prepare_dataset
# # #
# # #     prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
# # # print("BPE Tokenizer is ready.")
# # # #
# # # # print("Creating RobertaTokenizerFast.")
# # # # if not isfile(args.roberta_fast_tokenizer_path + "/merges.txt"):
# # # #     import roberta_tokenizer
# # # #
# # # #     roberta_tokenizer.save_roberta_tokenizer(path=args.bpe_path, save_to=args.roberta_fast_tokenizer_path)
# # # # print("RobertaFastTokenizer is ready.")
# # # #
# # # import yaml
# # # import WIP_BartSettings
# # #
# # # with open(args.hyperparameters_path) as file:
# # #     hyperparameters = yaml.safe_load(file)
# # #     for key in hyperparameters.keys():
# # #         print("Starting pretraining with {} parameter set.".format(key))
# # #         # TODO: INTEGRATE THIS MORE! Got to be able to call them to create the file...
# # #         WIP_BartSettings.train_and_save_BART(hyperparameters_dict=hyperparameters[key],
# # #                                                    selfies_path=args.prepared_data_path,
# # #                                                    bpe_path=args.bpe_path,
# # #                                                    save_to="./saved_models/" + key + "_saved_model/")
# # #         print("Finished pretraining with {} parameter set.\n---------------\n".format(key))
# # #
# # # # Check to see
# # # # def create_selfies_file(selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000, do_subset=True):
# # # #     selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling
# # # #
# # # #     if do_subset:
# # # #         selfies_subset = selfies_df.selfies[:subset_size]
# # # #     else:
# # # #         selfies_subset = selfies_df.selfies
# # # #     selfies_subset = selfies_subset.to_frame()
# # # #     selfies_subset["selfies"].to_csv(save_to, index=False, header=False)
# # # #
# # # # # create
# #
# # import argparse
# # import pandas as pd
# # import yaml
# # import WIP_BartSettings
# #
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
# # parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
# # parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
# # parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
# # parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
# # parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
# # args = parser.parse_args()
# #
# # try:
# #     df = pd.read_csv(args.selfies_dataset)
# # except FileNotFoundError:
# #     from prepare_dataset import prepare_dataset_for_pretrain
# #     print("No SEFLIES dataset")
# #     prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
# #     df = pd.read_csv(args.selfies_dataset)
# # print("We have a SELFIES set for ya!")
# #
# # print("Creating SELFIES.txt for tokenization.")
# # from os.path import isfile  # returns True if the file exists else False.
# #
# # if not isfile(args.prepared_data_path):
# #     from prepare_dataset import create_selfies_file
# #
# #     if args.subset_size != 0:
# #         create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
# #     else:
# #         create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
# # print("SELFIES .txt is ready for tokenization.")
# #
# # print("Creating BPE tokenizer.")
# # if not isfile(args.bpe_path + "/merges.txt"):
# #     import prepare_dataset
# #
# #     prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
# # print("BPE Tokenizer is ready.")
# #
# # with open(args.hyperparameters_path) as file:
# #     hyperparameters = yaml.safe_load(file)
# #
# # print("Starting pretraining with HIDDEN_SIZE parameter set.")
# # WIP_BartSettings.train_and_save_BART(
# #     hyperparameters_dict=hyperparameters,
# #     selfies_path=args.prepared_data_path,
# #     bpe_path=args.bpe_path,
# #     save_to="./saved_models/BART_saved_model/"
# # )
# # print("Finished pretraining with HIDDEN_SIZE parameter set.\n---------------\n")


# # START: train_for_pretrain.py
# ### end

# # add some checks
# import os

# import argparse
# import pandas as pd
# import yaml
# import WIP_BartSettings

# parser = argparse.ArgumentParser()
# parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
# parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
# parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
# parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
# parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
# parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
# args = parser.parse_args()

# # TODO: Modify bpe_path and prepared_data_path to be based off of the yaml key
# # This should be done by iterating over the keys in the hyperparameters dictionary and then setting the bpe_path and prepared_data_path to the key value.
# # This should be relatively simple! Then I can start training NNs on different datasets and hyperparameters.
# # WOOT WOOT...

# # BART_saved_model
# # Dir for pre-trained model is:
# # "./saved_models/BART_saved_model/"

# # if os.path.exists("./saved_models/BART_saved_model/"):
# #     print("Model directory exists.")
# # else:
# #     os.makedirs("./saved_models/BART_saved_model/")
# #     print("Created model directory.")

# # if not os.path.exists(args.prepared_data_path):
# #     os.makedirs(args.prepared_data_path)
# #     print(f"Created directory: {args.prepared_data_path}")
    
# # elif not os.path.isdir(args.prepared_data_path):
# #     raise NotADirectoryError(f"{args.prepared_data_path} is not a directory.")


# # TODO: Make sure that we still train the pre-train model if it doesn't exist.....
# # if not exist then we create the original pre-train else we skip it and go to fine-tuning....
# pretrain_check = os.listdir("./saved_models/BART_pretrained/") 
# if len(pretrain_check) == 0:
#     print("No pre-trained model found. Starting pre-training.")
# else:
#     print("Pre-trained model found. Skipping pre-training.")
    
# # NEED To integrate the above with the below...
# # if not os.path.exists(args.prepared_data_path):
# #     os.makedirs(args.prepared_data_path)
# #     print(f"Created directory: {args.prepared_data_path}")
# # elif not os.path.isdir(args.prepared_data_path):
# #     raise NotADirectoryError(f"{args.prepared_data_path} is not a directory.")



# try:
#     df = pd.read_csv(args.selfies_dataset)
# except FileNotFoundError:
#     from prepare_dataset import prepare_dataset_for_pretrain
#     print("No SEFLIES dataset")
#     prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
#     df = pd.read_csv(args.selfies_dataset)
# print("We have a SELFIES set for ya!")

# # TODO: Need to apply the selfies translation somewhere here....

# print("Creating SELFIES.txt for tokenization.")
# from os.path import isfile  # returns True if the file exists else False.

# if not isfile(args.prepared_data_path):
#     from prepare_dataset import create_selfies_file

#     if args.subset_size != 0:
#         create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
#     else:
#         create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path) # prepared_data_path should also be configurable to be from the yaml? 
# print("SELFIES .txt is ready for tokenization.")

# print("Creating BPE tokenizer.")
# if not isfile(args.bpe_path + "/merges.txt"):
#     import prepare_dataset
#     # TODO: This makes the bpe_tokenizer... but I do think in the prior file processing step. Maybe I need to separate this into this?
#     prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
# print("BPE Tokenizer is ready.")

# # TODO: NEW I think this is where the if check works... we check here and see if it exists?
# # We need to list a for-list with a YAML file that has multiple models in it... then we can iterate over the keys and train each model...
# # This includes not only just the pre-training but also the fine-tuning... NOT JUST BART...
# # RENAME BART to BART_pretrain as as the key....

# # The pretrain would be the base for each fine-tuning...


# # if model_name...

# # This hyperparameters path will contain lots of things and all the models... so this isn't part of the if statement...
# with open(args.hyperparameters_path) as file:
#     hyperparameters = yaml.safe_load(file)

# print("Loaded hyperparameters:", hyperparameters)  # Debug print to check the loaded hyperparameters



# # Access the 'BART' key in the hyperparameters dictionary
# # TODO: Modify this so then I can pass in multiple key's to create various models... The key name will be the model name directory folder name.
# # Edit further to allow for multiple models to be trained in job submission. This would ideally set up and redefine the vocabulary for each model.
# # This would be done by passing in a list of keys to the hyperparameters dictionary but would probably need to include additional parameters such as
# # molecular weight, WHAT ELSE DID I UTILIZE IN MY SQL SCRIPT? I should be able to target particularly pertinent subsets of ChEMBL or other datasets...

# bart_hyperparameters = hyperparameters.get("BART", {}) # This would have some model specific hyperparameters and probably be part of a for loop to iterate over the keys representing each model in the hyperparameters dictionary.
# for key in bart_hyperparameters.keys():
#     print("Starting pretraining with HIDDEN_SIZE parameter set.")
#     WIP_BartSettings.train_and_save_BART(
#         hyperparameters_dict=key,
#         selfies_path=args.prepared_data_path, # UPDATE this should be based off of the yaml key... I think...
#         bpe_path=args.bpe_path, # UPDATE this should be based off of the yaml key as well...
#         save_to=f"./saved_models/BART_saved_model_{key}/" # This would utilize an f-string for the "BART_saved_model" so then I can get multiple saved models...
#     )
# print("Finished pretraining with HIDDEN_SIZE parameter set.\n---------------\n")


# """
# TODO: Figure out if I only need to pretrain once or if I need to do it for each model... I think I only need to do it once but I need to check that. 
# Then I can fine-tune them all from there on subsets of data?

# I would then have to parse if the `save_to` is pre-train or fine-tune and then adjust the training accordingly so it would go pre-train then filter and fine-tune....
# this would allow me to do a model then the next one... then I can come back later and do molecular generation tasks...


# TODO: Also need to work to add the fine tuning on that particular dataset....
# Ideally the pretraining would include all the molecules but then the fine tuning would be done on a subset of the data.
# This is where the MW and other factors would play into the model training phase. I could then 



# TODO: This would involve adjusting the python script that does the SQL load in to include the molecular weight (and other factors) that I can feed into the sql query 
# to save that subset then put into the fine-tune script....
# """
 
# # END: train_for_pretrain.py

import os
import argparse
import pandas as pd
import yaml
from os.path import isfile
import WIP_BartSettings
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies

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
    # create the read in model_name_{key}.csv file
    # Convert to SELFIES
    # create_selfies = pd.read_csv(f"model_name_{key}.csv")
    # create_selfies["selfies"] = create_selfies["canonical_smiles"]
    # create_selfies.selfies = create_selfies.selfies.parallel_apply(convert_to_selfies)
    # df.drop(df[df.canonical_smiles == df.selfies].index, inplace=True)
    # df.drop(columns=["canonical_smiles"], inplace=True)
    # create_selfies_save_to = f"./data/trainable_selfies_{key}.csv"
    # create_selfies.to_csv(create_selfies_save_to, index=False)
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

def train_model(hyperparameters, args, key):
    print(f"Starting pretraining with {key} parameter set.")
    WIP_BartSettings.train_and_save_BART(
        hyperparameters_dict=hyperparameters[key],
        selfies_path=f"./data/trainable_selfies_{key}.csv",
        bpe_path=args.bpe_path,
        save_to=f"./saved_models/{key}_saved_model/"
    )
    print(f"Finished pretraining with {key} parameter set.\n---------------\n")

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

        # NEED TO TOKENIZE without SELFIES header up top and then feed in with selfies...
        # I ALSO Need to process to add in the inhibition site again? I think?
        # This is the confusing part....
        
        # TODO: I also need to get IC50 values and inhibition_site....
        # FIX: where is my query for that?
        #        bs.site_name, in data_needed.sql is where the thingy is...
        # then IC50 is also in relation?... idk
        
        # Prepare data
        prepare_data(args, key)

        # Train model
        train_model(bart_hyperparameters, args, key)

if __name__ == "__main__":
    main()
    
    
    
"""
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig, AdamW
from tqdm import tqdm

# Load the tokenizer and dataset
tokenizer = Tokenizer.from_file("./data/bpe_non/bpe.json")
fine_tune_dataset = SelfiesDataset(csv_file='./ChEMBL34_druglike_activity.csv', tokenizer_path='./data/bpe_non/bpe.json', mode='finetune')

# Create DataLoader
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Initialize BART model
config = BartConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_position_embeddings=1024,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=12,
    decoder_attention_heads=12,
    encoder_ffn_dim=3072,
    decoder_ffn_dim=3072,
    hidden_size=1152,
    pad_token_id=tokenizer.token_to_id("<pad>"),
    bos_token_id=tokenizer.token_to_id("<s>"),
    eos_token_id=tokenizer.token_to_id("</s>")
)
model = BartForConditionalGeneration(config)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(10):  # Number of epochs
    total_loss = 0
    for batch in tqdm(fine_tune_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass (assuming self-supervised learning, labels = input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Compute loss and optimize
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(fine_tune_loader)}")

# Save the model
torch.save(model.state_dict(), './selfies_BART_finetuned.pth')
"""
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
"""