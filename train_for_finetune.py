import argparse
import pandas as pd
import numpy as np
from os.path import isfile
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies
from SelfiesDataHandler import SelfiesDataset, collate_fn_fine, NNLossHandler, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer
import torch
from tqdm import tqdm  
import csv # FOR SAVING THE LOSS
from utils import diff_to_string, encode_differences_to_string, save_final_model_if_needed, write_done_marker, make_optimizer, make_scheduler
from pytorch_lamb import Lamb
from torch.nn.utils import clip_grad_norm_
import os
from utils import diff_to_string, encode_differences_to_string, save_final_model_if_needed, write_done_marker, make_optimizer, make_scheduler, load_hyperparameters
from train_for_pretrain import ClusteredSelfiesDataset

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
        
def train_for_finetune(model, train_loader, val_loader, cfg, save_dir, csv_file_path):
    """
    model        : a BartForConditionalGeneration
    finetune_train_loader : DataLoader for finetune set
    val_loader   : DataLoader for validation set
    cfg          : one of your hyperparameter dicts (e.g. hyperparameters_dict[key])
    save_dir     : where to save the best model
    csv_file_path: writes to the designated csv_file_path
    """
    os.makedirs(save_dir, exist_ok=True)
    # csv_path = os.path.join(save_dir, "training_log.csv")
    # csv_file = open(csv_path, "w", newline="")
    csv_file = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader), cfg["TRAIN_EPOCHS"])

    best_val_loss = float('inf')
    # Check for checkpoint
    checkpoint_path = os.path.join(save_dir, "checkpoint_resume.pt")
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint["best_val_loss"]
        patience = checkpoint["patience"]
        start_epoch = checkpoint["epoch"] + 1
    else:
    # Back to config setup
        patience = 0
    thresh = cfg["early_stopping_threshold"]
    max_patience = cfg["early_stopping_patience"]

    for epoch in range(start_epoch, cfg["TRAIN_EPOCHS"] + 1):
        # Training
        model.train()
        total_train_loss = 0.0
        # for batch in train_loader:
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            # for batch in val_loader:
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask'],
                             labels=batch['input_ids']).loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Save if new best model
        if avg_val_loss < best_val_loss - thresh:
            best_val_loss = avg_val_loss
            patience = 0
            model.save_pretrained(save_dir)

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_val_loss": best_val_loss,
                "patience": patience,
            }
            torch.save(checkpoint, os.path.join(save_dir, "checkpoint_resume.pt"))
            print(f"✅ Checkpoint saved at epoch {epoch}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"? Early stopping (no improvement in {max_patience} epochs)")
                break


        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        #     Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2E}")
        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{current_lr:.2E}"])
        csv_file.flush()

    # close the CSV file now that training (or early stop) is done
    csv_file.close()

    # Final model save (if needed) + mark training complete
    save_final_model_if_needed(model, save_dir)
    write_done_marker(save_dir)

def finetune_BART(hyperparameters_dict, args, key):
    # Setup File config parameters
    base_model_name = "skip_base"  # Customize as needed
    base_config = hyperparameters_dict[base_model_name]
    current_config = hyperparameters_dict[key]

    # diffs = diff_to_string(base_config, current_config)
    filename_stub = encode_differences_to_string(base_model_name, base_config, current_config)
    # filename_stub = encode_differences_to_string(base_model_name, diffs)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./selfies_word_tokenizer")
    model_save_dir = f'./selfies_BART_pretrained__{filename_stub}'
    csv_file_path = f'./finetuning_loss__{filename_stub}.csv'

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

    # finetune_train_dataset = ClusteredSelfiesDataset(df=df_train, tokenizer=tokenizer,
    #                                         mode="finetune")
    # finetune_validation_dataset = ClusteredSelfiesDataset(df=df_val, tokenizer=tokenizer,
    #                                              mode="finetune")
    finetune_train_dataset = SelfiesDataset(dataframe=df_train, tokenizer=tokenizer,
                                            mode="finetune")
    finetune_validation_dataset = SelfiesDataset(dataframe=df_val, tokenizer=tokenizer,
                                                 mode="finetune")


    # TODO: NEW 02/18/2025 get the IC50 finetune data!

    # TODO: Above should be the one fine_tune approach. It is the same for all models...
    # TODO: NEW 02/18/2025... This should be a load the finetune train and validation models and randomly grab 90/10% split...
    #

    # Create DataLoader
    finetune_train_loader = DataLoader(
        finetune_train_dataset, batch_size=8, shuffle=True,
        collate_fn=lambda x: collate_fn(x, mode='fine')
    )
    finetune_validation_loader = DataLoader(
        finetune_validation_dataset, batch_size=8, shuffle=True,
        collate_fn=lambda x: collate_fn(x, mode='fine')
    )

    model_path = "DataForGen/selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model config vocab size:", model.config.vocab_size)
    print(tokenizer.special_tokens_map)

    train_for_finetune(model, finetune_train_loader, finetune_validation_loader, current_config, model_save_dir, csv_file_path)


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
        if key.startswith("skip_"):
            continue

        # if os.path.exists(f'./selfies_BART_pretrained_{key}.pth'):
        #     print("Model already exists! No need to retrain")
        # Before training starts:
        filename_stub = encode_differences_to_string("skip_base", bart_hyperparameters["skip_base"],
                                                     bart_hyperparameters[key])
        model_save_dir = f'./selfies_BART_finetuned__{filename_stub}'

        # if os.path.exists(model_save_dir):
        #     print(f"✅ Model for '{key}' already exists at {model_save_dir}. Skipping...")
        done_flag = os.path.join(model_save_dir, "done.txt")

        if os.path.exists(done_flag):
            print(f"✅ Model '{key}' already completed (done.txt found) — skipping retrain.")
            continue
        else:
            args.smiles_dataset = f"model_name_{key}.csv"
            print(args.smiles_dataset)
            args.selfies_dataset = f"./data/molecule_data_{key}.csv"
            args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"
            # args.bpe_path = f"./data/bpe_filter_{key}/"

            # prepare_data(args, key)
            finetune_BART(bart_hyperparameters, args, key)


if __name__ == "__main__":
    main()
    
"""
TODO: Finetuning data requires the specific dataset from the beginning...
it needs to have site_name for it... My pretrain data does NOT need it...
"""

# TODO: I need to add early_stopping to the hyperparameters...