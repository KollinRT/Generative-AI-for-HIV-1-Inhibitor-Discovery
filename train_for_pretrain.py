import argparse
import csv
import itertools
import os
from os.path import isfile
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    List,
    Iterator,
    TypedDict,
)

import dask.dataframe as dd
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    PreTrainedTokenizerFast,
)

from SelfiesDataHandler import collate_fn, SelfiesIterableDataset, SelfiesDataset, SelfiesDatasetRound2
from utils import (
    save_final_model_if_needed,
    write_done_marker,
    make_optimizer,
    load_hyperparameters,
    make_scheduler_steps,
    make_scheduler
)

# BACKUP_EVERY_BATCH = 400
BACKUP_EVERY_BATCH = 10000

gpu_used = "B200"


# gpu_used = "4090"


class BatchDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TrainConfig(TypedDict, total=False):
    # training / validation cadence
    MAX_TRAINING_STEPS: int
    VAL_EVERY_STEPS: int
    MAX_VALID_BATCH_SIZE: Optional[int]

    # early stopping
    early_stopping_threshold: float
    early_stopping_patience: int
    early_stopping_toggle: bool

    # dataloader sizes
    TRAIN_BATCH_SIZE: int
    VALID_BATCH_SIZE: int

    # model config
    MAX_POSITION_EMBEDDINGS: int
    ENCODER_LAYERS: int
    DECODER_LAYERS: int
    NUM_ENCODER_ATTENTION_HEADS: int
    NUM_DECODER_ATTENTION_HEADS: int
    ENCODER_FFN_DIM: int
    DECODER_FFN_DIM: int
    HIDDEN_SIZE: int
    DROPOUT: float


def run_validation_batched(
    model: BartForConditionalGeneration,
    epoch: int,
    val_loader: DataLoader[BatchDict],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_val_loss: float = 0.0
    batches_seen: int = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
            total_val_loss += loss.item()
            batches_seen += 1

            if max_batches and batches_seen >= max_batches:
                break

    # avg_val_loss = total_val_loss / batches_seen
    avg_val_loss: float = total_val_loss / batches_seen
    return avg_val_loss

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
        # Training
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


def train_for_pretrain_steps(
    model: BartForConditionalGeneration,
    train_loader: DataLoader[BatchDict],
    val_loader: DataLoader[BatchDict],
    cfg: TrainConfig,
    save_dir: str,
    csv_file_path: str,
) -> None:
    """
    model        : a BartForConditionalGeneration
    train_loader : DataLoader for pretrain set
    val_loader   : DataLoader for validation set
    cfg          : one of your hyperparameter dicts (e.g. hyperparameters_dict[key])
    save_dir     : where to save the best model
    csv_file_path: writes to the designated csv_file_path
    """
    # Hyperparameters
    max_training_steps: int = cfg["MAX_TRAINING_STEPS"]
    validate_every_steps: int = cfg["VAL_EVERY_STEPS"]
    early_stopping_threshold: float = cfg["early_stopping_threshold"]
    max_patience: int = cfg["early_stopping_patience"]
    max_valid_batches: Optional[int] = cfg.get("MAX_VALID_BATCH_SIZE")

    # Setup 
    train_iterator: Iterator[BatchDict] = itertools.cycle(
        train_loader
    )  # Infinite looping
    os.makedirs(save_dir, exist_ok=True)

    # Safe CSV setup (no overwrite)
    write_header: bool = (
        not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0
    )
    csv_file = open(csv_file_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(
            ["epoch", "global_step", "train_loss", "val_loss", "learning_rate"]
        )

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)
    model.to(device)

    optimizer: torch.optim.Optimizer = make_optimizer(model, cfg)  # type: ignore[arg-type]
    print(f"length of train_loader: {len(train_loader)}")

    scheduler = make_scheduler_steps(optimizer, cfg, max_training_steps)

    # Checkpoint Resume 
    # Check for checkpoint
    checkpoint_path: str = os.path.join(save_dir, "checkpoint_resume.pt")
    global_step: int = 0
    best_val_loss: float = float("inf")

    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path)
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = float(checkpoint["best_val_loss"])
        patience = int(checkpoint["patience"])
        global_step = int(checkpoint["step"]) + 1

        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                last_row = list(csv.reader(f))[-1]
                last_logged_step = int(
                    last_row[1]
                )  # assuming 'global_step' is column index 1
                if global_step <= last_logged_step:
                    print(
                        f"⚠️ Warning: Resuming at step {global_step} but last logged step is {last_logged_step}."
                    )
                    print(
                        "🧹 You might want to clean or truncate the CSV to prevent mixing runs."
                    )
    else:
        # Back to config setup
        patience = 0

    # Mixed Precision 
    use_amp: bool = gpu_used == "B200"
    scaler: Optional[GradScaler] = GradScaler("cuda") if use_amp else None

    total_train_loss: float = 0.0
    step_in_epoch: int = 0

    # Main Training Loop
    with tqdm(
        total=max_training_steps,
        initial=global_step,
        desc="Training",
        dynamic_ncols=True,
    ) as pbar:
        while global_step < max_training_steps:
            # Training
            model.train()

            # One batch
            batch: BatchDict = next(train_iterator)
            batch = {k: v.to(device) for k, v in batch.items()}  # type: ignore[assignment]

            #  Forward / Backward pass
            if use_amp:
                assert scaler is not None
                with autocast("cuda"):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            total_train_loss += float(loss.item())
            global_step += 1
            step_in_epoch += 1

            # Potential way to prevent a memory leak
            del batch, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation, Logging, Checkpointing
            if global_step % validate_every_steps == 0:
                avg_train_loss: float = total_train_loss / step_in_epoch

                avg_val_loss: float = run_validation_batched(
                    model,
                    global_step,
                    val_loader,
                    device,
                    max_batches=max_valid_batches,
                )

                # Save best model
                if avg_val_loss < best_val_loss - early_stopping_threshold:
                    best_val_loss = avg_val_loss
                    patience = 0
                    model.save_pretrained(save_dir)
                    torch.save(
                        {
                            "step": global_step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict()
                            if scheduler
                            else None,
                            "best_val_loss": best_val_loss,
                            "patience": patience,
                        },
                        checkpoint_path,
                    )
                    print(f"✅ Checkpoint saved at step {global_step}")
                else:
                    patience += 1
                    if cfg.get("early_stopping_toggle", True):
                        if patience >= max_patience:
                            print(
                                f"⛔ Early stopping triggered after {max_patience} validations with no improvement."
                            )
                            csv_file.close()
                            save_final_model_if_needed(
                                model, save_dir, optimizer, scheduler
                            )
                            write_done_marker(save_dir)
                            return

                # Steps for mini-backup...
                # Mini-backup every N validation steps
                if (
                    global_step
                    % (
                        validate_every_steps
                        * BACKUP_EVERY_BATCH
                        // validate_every_steps
                    )
                    == 0
                ):
                    backup_path = os.path.join(
                        save_dir, f"backup_step_{global_step}.pt"
                    )
                    torch.save(
                        {
                            "step": global_step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict()
                            if scheduler
                            else None,
                            "best_val_loss": best_val_loss,
                            "patience": patience,
                        },
                        backup_path,
                    )
                    print(
                        f"💾 Backup checkpoint saved at step {global_step} → {backup_path}"
                    )

                current_lr: float = float(optimizer.param_groups[0]["lr"])
                # Guard against datasets that don't implement __len__
                dataset_len = getattr(train_loader.dataset, "__len__", None)
                if callable(dataset_len):
                    ds_len_val = len(train_loader.dataset)  # type: ignore[arg-type]
                    pseudo_epoch = (global_step * train_loader.batch_size) // ds_len_val
                else:
                    pseudo_epoch = 0

                print(
                    f"[Epoch {pseudo_epoch} | Step {global_step}] train_loss={avg_train_loss:.4f}  "
                    f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2E}"
                )
                csv_writer.writerow(
                    [
                        pseudo_epoch,
                        global_step,
                        f"{avg_train_loss:.6f}",
                        f"{avg_val_loss:.6f}",
                        f"{current_lr:.2E}",
                    ]
                )
                csv_file.flush()

                # Reset counters after validation
                total_train_loss = 0.0
                step_in_epoch = 0
                pbar.update(1)  # update tqdm bar

    # Training Complete 
    csv_file.close()

    # Save final model separately
    model.save_pretrained(os.path.join(save_dir, f"final_step_{global_step}"))

    # Save done marker in root save_dir
    write_done_marker(save_dir)

    # Save final model separately
    final_model_dir: str = os.path.join(save_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    print(f"💾 Final model saved to {final_model_dir}")

    # Mark training as complete
    write_done_marker(save_dir)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
        },
        os.path.join(final_model_dir, "final_checkpoint.pt"),
    )


def prepare_data(args: argparse.Namespace, key: str) -> None:
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
        prepare_dataset_for_pretrain(
            path=args.smiles_dataset, save_to=args.selfies_dataset
        )
        df = pd.read_csv(args.selfies_dataset)
    print("We have a SELFIES set for ya!")

    print("Creating SELFIES.txt for tokenization.")
    if not isfile(args.prepared_data_path):
        from prepare_dataset import create_selfies_file

        if args.subset_size != 0:
            create_selfies_file(
                df,
                subset_size=args.subset_size,
                do_subset=True,
                save_to=args.prepared_data_path,
            )
        else:
            create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
    print("SELFIES .txt is ready for tokenization.")

    print("Creating file for training!")
    if not isfile(f"./data/trainable_selfies_{key}.csv"):
        from prepare_dataset import prepare_dataset_for_pretrain

        prepare_dataset_for_pretrain(
            f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv"
        )
    print(f"File for training is ready! (trainable_selfies_{key}.csv)")




# def pretrain_BART(
#     hyperparameters_dict: Mapping[str, TrainConfig],
#     args: argparse.Namespace,
#     key: str,
# ) -> None:
#     """
#     Args:
#         hyperparameters_dict: YAML file to be passed in with configuration.
#         args: NOT NECESSARY.
#         key: key name of model for downstream naming.

#     Returns:

#     """
#     # Setup File config parameters
#     current_config = hyperparameters_dict[key]

#     run_dir: str = f"./runs/selfies_BART_PRETRAIN_{key}"
#     model_save_dir: str = os.path.join(run_dir, "model")
#     csv_file_path: str = os.path.join(run_dir, "pretraining_loss.csv")

#     # Define Training Hyperparameters
#     train_batch_size: int = current_config["TRAIN_BATCH_SIZE"]
#     val_batch_size: int = current_config["VALID_BATCH_SIZE"]

#     # Load the tokenizer
#     # tokenizer = PreTrainedTokenizerFast.from_pretrained("./selfies_word_tokenizer")
#     tokenizer = PreTrainedTokenizerFast.from_pretrained(
#         "./code_run_files/full_tokenizer_finetune_and_pretrain"
#     )

#     df = pd.read_csv(f"model_name_{key}.csv")
#     # print(df.columns)
#     # print(df.head(2))

#     # Sort by cluster and split into training and validation sets
#     # df = df.sort_values(by=['Cluster'])
#     # mol_count = int(len(df) * 0.9)
#     # Split 90% train, 10% validation
#     # train_frac = 0.9
#     # df = df.shuffle(random_state=42, shuffle="tasks")
#     # train_df = df[:mol_count]  # 90% for training
#     # valid_df = df[mol_count:]  # 10% for validation
#     # train_df = df.sample(frac=train_frac, random_state=42)
#     # valid_df = df.drop(train_df.index)
#     # train_frac = 0.9
#     # train_df, valid_df = df.random_split([train_frac, 1 - train_frac], random_state=42)
#     train_frac = 0.9
#     train_df = df.sample(frac=train_frac, random_state=42).reset_index(drop=True)
#     valid_df = df.drop(train_df.index).reset_index(drop=True)

#     # randomize the data and redo it.
#     # train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     # valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

#     # # Instead of passing the raw DataFrame to DataLoader, wrap it in the new ClusteredSelfiesDataset
#     # Partition-based split for iterable dataset
#     n_partitions: int = df.npartitions
#     train_partitions: List[int] = list(range(int(n_partitions * 0.9)))  # first 90%
#     val_partitions: List[int] = list(
#         range(int(n_partitions * 0.9), n_partitions)
#     )  # last 10%

#     train_dataset = SelfiesDataset(
#         df, tokenizer, mode="pretrain"
#     )
#     valid_dataset = SelfiesDataset(
#         df, tokenizer, mode="pretrain"
#     )

#     use_amp: bool = gpu_used == "B200"
#     if use_amp:
#         # B200
#         pretrain_loader = DataLoader(
#             train_dataset,
#             batch_size=train_batch_size,
#             collate_fn=lambda x: collate_fn(x, mode="pre"),
#             num_workers=0,
#             pin_memory=True,
#         )
#         val_loader = DataLoader(
#             valid_dataset,
#             batch_size=val_batch_size,
#             collate_fn=lambda x: collate_fn(x, mode="pre"),
#             num_workers=0,
#             pin_memory=True,
#         )
#     else:
#         # 4090
#         pretrain_loader = DataLoader(
#             train_dataset,
#             batch_size=train_batch_size,
#             collate_fn=lambda x: collate_fn(x, mode="pre"),
#         )
#         val_loader = DataLoader(
#             valid_dataset,
#             batch_size=val_batch_size,
#             collate_fn=lambda x: collate_fn(x, mode="pre"),
#         )

#     config = BartConfig(
#         vocab_size=tokenizer.vocab_size,
#         max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],
#         encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"],
#         decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
#         encoder_attention_heads=hyperparameters_dict[key][
#             "NUM_ENCODER_ATTENTION_HEADS"
#         ],
#         decoder_attention_heads=hyperparameters_dict[key][
#             "NUM_DECODER_ATTENTION_HEADS"
#         ],
#         encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
#         decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
#         hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],
#         pad_token_id=tokenizer.pad_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         mask_token_id=tokenizer.mask_token_id,
#     )
#     model = BartForConditionalGeneration(config)
#     # model = torch.compile(model)

#     # # Print DataFrame info for debugging
#     # print("Final training DataFrame:")
#     # print(train_df.head(2))

#     # Training Loop START
#     # train_for_pretrain(model, pretrain_loader, val_loader, current_config, model_save_dir, csv_file_path)
#     train_for_pretrain(
#         model,
#         pretrain_loader,
#         val_loader,
#         current_config,
#         model_save_dir,
#         csv_file_path,
#     )

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
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./code_run_files/full_tokenizer_finetune_and_pretrain")

    # Load DF and Cluster
    # Load and process the DataFrame: apply fingerprinting and clustering
    # df = pd.read_csv(f"./data/trainable_selfies_{key}_FP_CLUSTERED_256perms_7_clustered.csv")
    df = pd.read_csv(f"model_name_{key}_selfies.csv")
    print(df.columns)
    print(df.head(2))

    # Sort by cluster and split into training and validation sets
    # df = df.sort_values(by=['Cluster'])
    mol_count = int(len(df) * 0.9)
    train_df = df[:mol_count]  # 90% for training
    valid_df = df[mol_count:]  # 10% for validation

    # randomize the data and redo it.
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Instead of passing the raw DataFrame to DataLoader, wrap it in the new ClusteredSelfiesDataset
    train_dataset = SelfiesDatasetRound2(train_df, tokenizer, mode='pretrain')
    valid_dataset = SelfiesDatasetRound2(valid_df, tokenizer, mode='pretrain')

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


def pretrain_BART_steps(
    hyperparameters_dict: Mapping[str, TrainConfig],
    args: argparse.Namespace,
    key: str,
) -> None:
    """
    Args:
        hyperparameters_dict: YAML file to be passed in with configuration.
        args: NOT NECESSARY.
        key: key name of model for downstream naming.

    Returns:

    """
    # Setup File config parameters
    current_config = hyperparameters_dict[key]

    run_dir: str = f"./runs/selfies_BART_PRETRAIN_{key}"
    model_save_dir: str = os.path.join(run_dir, "model")
    csv_file_path: str = os.path.join(run_dir, "pretraining_loss.csv")

    # Define Training Hyperparameters
    train_batch_size: int = current_config["TRAIN_BATCH_SIZE"]
    val_batch_size: int = current_config["VALID_BATCH_SIZE"]

    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "./code_run_files/full_tokenizer_finetune_and_pretrain"
    )

    parquet_path: str = "./model_4_warmup_FP_pre.parquet"
    # Load DF and Cluster
    df = dd.read_parquet(parquet_path)

    # print(df.columns)
    # print(df.head(2))

    train_frac = 0.9
    train_df, valid_df = df.random_split([train_frac, 1 - train_frac], random_state=42)

    # randomize the data and redo it.
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # # Instead of passing the raw DataFrame to DataLoader, wrap it in the new ClusteredSelfiesDataset
    # Partition-based split for iterable dataset
    df = dd.read_parquet(parquet_path)
    n_partitions: int = df.npartitions

    train_partitions: List[int] = list(range(int(n_partitions * 0.9)))  # first 90%
    val_partitions: List[int] = list(
        range(int(n_partitions * 0.9), n_partitions)
    )  # last 10%

    train_dataset = SelfiesIterableDataset(
        parquet_path, tokenizer, partitions=train_partitions, mode="pretrain"
    )
    valid_dataset = SelfiesIterableDataset(
        parquet_path, tokenizer, partitions=val_partitions, mode="pretrain"
    )

    use_amp: bool = gpu_used == "B200"
    if use_amp:
        # B200
        pretrain_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=lambda x: collate_fn(x, mode="pre"),
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=val_batch_size,
            collate_fn=lambda x: collate_fn(x, mode="pre"),
            num_workers=0,
            pin_memory=True,
        )
    else:
        # 4090
        pretrain_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=lambda x: collate_fn(x, mode="pre"),
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=val_batch_size,
            collate_fn=lambda x: collate_fn(x, mode="pre"),
        )

    config = BartConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=hyperparameters_dict[key]["MAX_POSITION_EMBEDDINGS"],
        encoder_layers=hyperparameters_dict[key]["ENCODER_LAYERS"],
        decoder_layers=hyperparameters_dict[key]["DECODER_LAYERS"],
        encoder_attention_heads=hyperparameters_dict[key][
            "NUM_ENCODER_ATTENTION_HEADS"
        ],
        decoder_attention_heads=hyperparameters_dict[key][
            "NUM_DECODER_ATTENTION_HEADS"
        ],
        encoder_ffn_dim=hyperparameters_dict[key]["ENCODER_FFN_DIM"],
        decoder_ffn_dim=hyperparameters_dict[key]["DECODER_FFN_DIM"],
        hidden_size=hyperparameters_dict[key]["HIDDEN_SIZE"],
        dropout=hyperparameters_dict[key][
            "DROPOUT"
        ],  # TODO: 08/3/25 ADD AS A PARAMETER
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    model = BartForConditionalGeneration(config)

    # # Print DataFrame info for debugging
    # print("Final training DataFrame:")
    # print(train_df.head(2))

    # Training Loop START
    train_for_pretrain_steps(
        model,
        pretrain_loader,
        val_loader,
        current_config,
        model_save_dir,
        csv_file_path,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_dataset",
        required=False,
        metavar="/path/to/dataset/*.csv",
        help="Path of the SMILES dataset.",
    )
    parser.add_argument(
        "--selfies_dataset",
        required=False,
        metavar="/path/to/dataset/*.csv",
        help="Path of the SELFIES dataset.",
    )
    parser.add_argument(
        "--subset_size",
        required=False,
        metavar="<int>",
        type=int,
        default=0,
        help="Subset size to use (0 for full dataset).",
    )
    parser.add_argument(
        "--hyperparameters_path",
        required=True,
        metavar="/path/to/hyperparameters/",
        help="Path to hyperparameters YAML file.",
    )
    parser.add_argument(
        "--steps",
        action='store_true',
    )
    args = parser.parse_args()

    hyperparameters: Mapping[str, Any] = load_hyperparameters(args.hyperparameters_path)
    print(
        "Loaded hyperparameters:", hyperparameters
    ) 
    bart_hyperparameters = hyperparameters.get("BART", {})
    print("BART hyperparameters:", bart_hyperparameters)

    for key in bart_hyperparameters.keys():
        if key.startswith("skip_"):
            continue

        run_dir = f"./runs/selfies_BART_PRETRAIN_{key}"

        model_save_dir = os.path.join(run_dir, "model")
        done_flag = os.path.join(model_save_dir, "done.txt")

        if os.path.exists(done_flag):
            print(
                f"Model '{key}' already completed (done.txt found) — skipping retrain."
            )
            continue
        else:
            args.smiles_dataset = f"model_name_{key}.csv"
            print(args.smiles_dataset)
            args.selfies_dataset = f"./data/molecule_data_{key}.csv"
            # args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"

            # prepare_data(args, key)
            if args.steps:
                pretrain_BART_steps(bart_hyperparameters, args, key)
            else: 
                pretrain_BART(bart_hyperparameters, args, key)


if __name__ == "__main__":
    main()
