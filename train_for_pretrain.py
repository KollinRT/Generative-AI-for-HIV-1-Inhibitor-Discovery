import argparse
import csv
import os
from os.path import isfile

import pandas as pd
import selfies as sf
import torch
from pytorch_lamb import Lamb
from tokenizers import Tokenizer
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartConfig, PreTrainedTokenizerFast

from SelfiesDataHandler import SelfiesDataset, collate_fn
from utils import save_final_model_if_needed, write_done_marker, make_optimizer, make_scheduler, load_hyperparameters

# gpu_used = "B200"
gpu_used = "4090"


def train_for_pretrain(model, train_loader, val_loader, cfg, save_dir, csv_file_path):
    """
    model        : a BartForConditionalGeneration
    train_loader : DataLoader for pretrain set
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
    print(f"Device: {device}")
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

    use_amp = (gpu_used == "B200")
    scaler = GradScaler("cuda") if use_amp else None

    for epoch in range(start_epoch, cfg["TRAIN_EPOCHS"] + 1):
        # === Training ===
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with autocast("cuda"):
                    outputs = model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['input_ids'])
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

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


def prepare_data(args, key):
    """
    Args:
        args:
        key:

    Returns:

    """
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

    # TODO: 05/22/2025 Get encoding working here!
    # TODO: 05/23/2025 09:37:00 How does this fit in with padding?
    def __getitem__(self, idx):
        """Retrieve an item by index."""
        selfies_string = self.df.iloc[idx]['selfies']
        print("selfies_string:", selfies_string)
        # Correctly tokenize SELFIES using semantic splitting
        tokens = list(sf.split_selfies(selfies_string))  # ['[C]', '[C]', '[O]']
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

        print("input_ids:", input_ids)
        # Pad/truncate to max_length (e.g., 256)
        max_length = 256
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        print("padded input_ids:", input_ids)
        if self.mode == 'pretrain':
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        elif self.mode == 'finetune':
            IC50 = self.df.iloc[idx]['IC50']
            inhibition_site = self.df.iloc[idx]['site_name']
            inhibition_encoded = self.encode_inhibition_site(inhibition_site)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'IC50': torch.tensor([IC50], dtype=torch.float),
                'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
            }

    def encode_inhibition_site(self, inhibition_site):
        inhibition_site = inhibition_site.strip().upper()
        if 'RVP' in inhibition_site:
            return 0
        elif 'RVE' in inhibition_site:
            return 1
        return -1


# ===================================

def pretrain_BART(hyperparameters_dict, args, key):
    """
    Args:
        hyperparameters_dict: YAML file to be passed in with configuration.
        args: NOT NECESSARY.
        key: key name of model for downstream naming.

    Returns:

    """
    # Setup File config parameters
    base_model_name = "skip_base"  # Customize as needed
    base_config = hyperparameters_dict[base_model_name]
    current_config = hyperparameters_dict[key]

    run_dir = f"./runs/selfies_BART_PRETRAIN_{key}"
    model_save_dir = os.path.join(run_dir, "model")
    csv_file_path = os.path.join(run_dir, "pretraining_loss.csv")

    # Define Training Hyperparameters
    train_batch_size = current_config["TRAIN_BATCH_SIZE"]
    val_batch_size = current_config["VALID_BATCH_SIZE"]

    # Load the tokenizer
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("./selfies_word_tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./selfies_word_tokenizer_12M")

    # Load DF and Cluster
    # Load and process the DataFrame: apply fingerprinting and clustering
    # TODO 06/14/2025 @ 11:34 AM: MAKE SURE THIS DIRECTORY ALIGNS!
    df = pd.read_csv(f"./data/trainable_selfies_{key}_FP_CLUSTERED_256perms_7_clustered.csv")
    print(df.columns)
    print(df.head(2))

    # Sort by cluster and split into training and validation sets
    df = df.sort_values(by=['Cluster'])
    mol_count = int(len(df) * 0.9)
    train_df = df[:mol_count]  # 90% for training
    valid_df = df[mol_count:]  # 10% for validation

    # randomize the data and redo it.
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Instead of passing the raw DataFrame to DataLoader, wrap it in the new ClusteredSelfiesDataset
    train_dataset = SelfiesDataset(train_df, tokenizer, mode='pretrain')
    valid_dataset = SelfiesDataset(valid_df, tokenizer, mode='pretrain')

    use_amp = (gpu_used == "B200")
    if use_amp:
        # B200
        pretrain_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                     collate_fn=lambda x: collate_fn(x, mode='pre'),
                                     num_workers=8, pin_memory=True
                                     )
        val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True,
                                collate_fn=lambda x: collate_fn(x, mode='pre'),
                                num_workers=8, pin_memory=True
                                )
    # 4090
    else:
        pretrain_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                     collate_fn=lambda x: collate_fn(x, mode='pre')
                                     )
        val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True,
                                collate_fn=lambda x: collate_fn(x, mode='pre')
                                )

    config = BartConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],
        encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"],
        decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
        encoder_attention_heads=hyperparameters_dict[key]["NUM_ENCODER_ATTENTION_HEADS"],
        decoder_attention_heads=hyperparameters_dict[key]["NUM_DECODER_ATTENTION_HEADS"],
        encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
        decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
        hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id
    )
    model = BartForConditionalGeneration(config)
    # model = torch.compile(model)

    # Print DataFrame info for debugging
    print("Final training DataFrame:")
    print(train_df.head(2))

    # Training Loop START
    train_for_pretrain(model, pretrain_loader, val_loader, current_config, model_save_dir, csv_file_path)


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
    print("Loaded hyperparameters:",
          hyperparameters)  # TODO: NEW 02/16/2025 figure out why BART is empty in combined_config... I THINK IT WORKS... ✓✓
    bart_hyperparameters = hyperparameters.get("BART", {})
    print("BART hyperparameters:", bart_hyperparameters)

    for key in bart_hyperparameters.keys():
        if key.startswith("skip_"):
            continue

        run_dir = f"./runs/selfies_BART_PRETRAIN_{key}"

        done_flag = os.path.join(run_dir, "done.txt")

        if os.path.exists(done_flag):
            print(f"✅ Model '{key}' already completed (done.txt found) — skipping retrain.")
            continue
        else:
            args.smiles_dataset = f"model_name_{key}.csv"
            print(args.smiles_dataset)
            args.selfies_dataset = f"./data/molecule_data_{key}.csv"
            # args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"

            # prepare_data(args, key)
            pretrain_BART(bart_hyperparameters, args, key)


if __name__ == "__main__":
    main()
