import argparse
import csv  # FOR SAVING THE LOSS
import os
from os.path import isfile

import numpy as np
import pandas as pd
import selfies as sf
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

from SelfiesDataHandler import collate_fn, SelfiesFinetuneDataset
from utils import (
    save_final_model_if_needed,
    write_done_marker,
    make_optimizer,
    make_scheduler,
    load_hyperparameters,
)

from torch.amp import autocast, GradScaler

gpu_used = "B200"
# gpu_used = "4090"

use_amp = gpu_used == "B200"
scaler = GradScaler("cuda") if use_amp else None


# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# def freeze_bart_encoder_except_last(model, num_decoder_layers_to_train=2):
#     for name, param in model.named_parameters():
#         if "encoder" in name or f"decoder.layer.{8 - num_decoder_layers_to_train}" not in name:
#             param.requires_grad = False


# def freeze_bart_layers(model, num_decoder_layers_unfrozen=1):
#     """
#     Freezes all layers in the model except the final `num_decoder_layers_unfrozen` decoder layers and LM head.
#     """
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Unfreeze LM head
#     for param in model.lm_head.parameters():
#         param.requires_grad = True
#
#     # Unfreeze final decoder layers
#     total_decoder_layers = len(model.model.decoder.layers)
#     for i in range(total_decoder_layers - num_decoder_layers_unfrozen, total_decoder_layers):
#         for param in model.model.decoder.layers[i].parameters():
#             param.requires_grad = True
#
#     # Unfreeze layer norm in decoder (optional: helps stability)
#     # for param in model.model.decoder.layernorm.parameters():
#     #     param.requires_grad = True
#     for param in model.model.decoder.final_layer_norm.parameters():
#         param.requires_grad = True
#
#     # Print summary
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"🔒 Model frozen. Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
#
# def freeze_bart_layers(model, num_decoder_layers_unfrozen=1):
#     """
#     Freezes all layers in the model except the final `num_decoder_layers_unfrozen` decoder layers and LM head.
#     """
#
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Unfreeze LM head
#     for param in model.lm_head.parameters():
#         param.requires_grad = True
#
#     # Unfreeze final decoder layers
#     total_decoder_layers = len(model.model.decoder.layers)
#     for i in range(total_decoder_layers - num_decoder_layers_unfrozen, total_decoder_layers):
#         for param in model.model.decoder.layers[i].parameters():
#             param.requires_grad = True
#
#     # ✅ FIXED: Unfreeze decoder final layer norm (if desired)
#     for param in model.model.decoder.final_layer_norm.parameters():
#         param.requires_grad = True
#
#     # Optional: Also unfreeze shared embedding (less common during finetuning)
#     # for param in model.model.shared.parameters():
#     #     param.requires_grad = True
#
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"🔒 Model frozen. Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
def freeze_bart_layers(model, num_decoder_layers_unfrozen=1):
    """
    Freezes all layers in the model except:
    - The last `num_decoder_layers_unfrozen` decoder layers
    - The LM head
    - The decoder final layer norm (if present)
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LM head
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreeze final N decoder layers
    total_decoder_layers = len(model.model.decoder.layers)
    for i in range(
        total_decoder_layers - num_decoder_layers_unfrozen, total_decoder_layers
    ):
        for param in model.model.decoder.layers[i].parameters():
            param.requires_grad = True

    # ✅ SAFE: Unfreeze decoder final layer norm if it exists
    if hasattr(model.model.decoder, "final_layer_norm"):
        for param in model.model.decoder.final_layer_norm.parameters():
            param.requires_grad = True

    # Debug print: how many parameters are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"🔒 Model frozen. Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


def prepare_data(args, key):  # TODO: Integrate key into here.... where?
    """
    Code to prepare data for training by handling command-line arguments
    Args:
        args:
        key:

    Returns:

    """
    try:
        df = pd.read_csv(args.selfies_dataset)
    except FileNotFoundError:
        from prepare_dataset import prepare_dataset_for_pretrain

        print("No SEFLIES dataset")
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
            )  # prepared_data_path is where the selfies by itself goes...
        else:
            create_selfies_file(
                df, do_subset=False, save_to=args.prepared_data_path
            )  # TODO: Need to ensure this is being done! LAST STEP!
    print("SELFIES .txt is ready for tokenization.")

    print("Creating file for training!")
    if not isfile(f"./data/trainable_selfies_{key}.csv"):
        from prepare_dataset import prepare_dataset_for_pretrain

        prepare_dataset_for_pretrain(
            f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv"
        )
    print(f"File for training is ready! (trainable_selfies_{key}.csv)")


# def train_for_finetune(model, train_loader, val_loader, cfg, save_dir, csv_file_path):
#     """
#     model        : a BartForConditionalGeneration
#     finetune_train_loader : DataLoader for finetune set
#     val_loader   : DataLoader for validation set
#     cfg          : one of your hyperparameter dicts (e.g. hyperparameters_dict[key])
#     save_dir     : where to save the best model
#     csv_file_path: writes to the designated csv_file_path
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     # csv_path = os.path.join(save_dir, "training_log.csv")
#     # csv_file = open(csv_path, "w", newline="")
#     csv_file = open(csv_file_path, "w", newline="")
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # DEBUGGING
#     # device = "cpu"
#     model.to(device)
#     print(f"Model vocab size: {model.model.shared.num_embeddings}")
#
#     optimizer = make_optimizer(model, cfg)
#     scheduler = make_scheduler(optimizer, cfg, len(train_loader), cfg["TRAIN_EPOCHS"])
#
#     best_val_loss = float('inf')
#     # Check for checkpoint
#     checkpoint_path = os.path.join(save_dir, "checkpoint_resume.pt")
#     start_epoch = 1
#     if os.path.exists(checkpoint_path):
#         print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint["model_state"])
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#         if scheduler and checkpoint.get("scheduler_state"):
#             scheduler.load_state_dict(checkpoint["scheduler_state"])
#         best_val_loss = checkpoint["best_val_loss"]
#         patience = checkpoint["patience"]
#         start_epoch = checkpoint["epoch"] + 1
#     else:
#         # Back to config setup
#         patience = 0
#     thresh = cfg["early_stopping_threshold"]
#     max_patience = cfg["early_stopping_patience"]
#
#     for epoch in range(start_epoch, cfg["TRAIN_EPOCHS"] + 1):
#         # Training
#         model.train()
#         total_train_loss = 0.0
#         # for batch in train_loader:
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             # Debug print
#             # print("🧪 [Train] input_ids stats:")
#             # print("  shape:", batch['input_ids'].shape)
#             # print("  max:", batch['input_ids'].max())
#             # print("  min:", batch['input_ids'].min())
#             # print("  dtype:", batch['input_ids'].dtype)
#             # print(f"batch['input_ids':\n{batch['input_ids']}")
#             outputs = model(input_ids=batch['input_ids'],
#                             attention_mask=batch['attention_mask'],
#                             labels=batch['input_ids'])
#             loss = outputs.loss
#             loss.backward()
#             clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
#                 scheduler.step()
#             optimizer.zero_grad()
#             total_train_loss += loss.item()
#
#         avg_train_loss = total_train_loss / len(train_loader)
#
#         # Validation
#         model.eval()
#         total_val_loss = 0.0
#         with torch.no_grad():
#             # for batch in val_loader:
#             for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 print("🧪 [Val] input_ids stats:")
#                 print("  shape:", batch['input_ids'].shape)
#                 print("  max:", batch['input_ids'].max())
#                 print("  min:", batch['input_ids'].min())
#                 print("  dtype:", batch['input_ids'].dtype)
#
#                 loss = model(input_ids=batch['input_ids'],
#                              attention_mask=batch['attention_mask'],
#                              labels=batch['input_ids']).loss
#                 total_val_loss += loss.item()
#
#         avg_val_loss = total_val_loss / len(val_loader)
#
#         # Save if new best model
#         if avg_val_loss < best_val_loss - thresh:
#             best_val_loss = avg_val_loss
#             patience = 0
#             model.save_pretrained(save_dir)
#
#             checkpoint = {
#                 "epoch": epoch,
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "scheduler_state": scheduler.state_dict() if scheduler else None,
#                 "best_val_loss": best_val_loss,
#                 "patience": patience,
#             }
#             torch.save(checkpoint, os.path.join(save_dir, "checkpoint_resume.pt"))
#             print(f"✅ Checkpoint saved at epoch {epoch}")
#         else:
#             patience += 1
#             if patience >= max_patience:
#                 print(f"? Early stopping (no improvement in {max_patience} epochs)")
#                 break
#
#         # Scheduler step
#         if scheduler:
#             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 scheduler.step(avg_val_loss)
#             elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
#                 scheduler.step()
#
#         #     Logging
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  "
#               f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2E}")
#         csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{current_lr:.2E}"])
#         csv_file.flush()
#
#     # close the CSV file now that training (or early stop) is done
#     csv_file.close()
#
#     # Final model save (if needed) + mark training complete
#     save_final_model_if_needed(model, save_dir)
#     write_done_marker(save_dir)


def train_for_finetune(model, train_loader, val_loader, cfg, save_dir, csv_file_path):
    os.makedirs(save_dir, exist_ok=True)
    csv_file = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader), cfg["TRAIN_EPOCHS"])

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(save_dir, "checkpoint_resume.pt")
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint["best_val_loss"]
        patience = checkpoint["patience"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        patience = 0

    thresh = cfg["early_stopping_threshold"]
    max_patience = cfg["early_stopping_patience"]

    # print(f"use_amp: {use_amp}")
    use_amp = False
    print(f"use_amp: {use_amp}")

    for epoch in range(start_epoch, cfg["TRAIN_EPOCHS"] + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            # # Debug
            # print(batch['input_ids'])
            # print(batch['attention_mask'])
            # print(batch['labels'])
            # labels = batch['labels']  # labels too if you want to check them
            # print(f"Labels unique values: {torch.unique(labels)}")

            batch = {k: v.to(device) for k, v in batch.items()}
            # for key in ['input_ids', 'attention_mask', 'labels']:
            #     if torch.isnan(batch[key]).any() or torch.isinf(batch[key]).any():
            #         print(f"Warning: {key} contains NaNs or Infs")

            # # optimizer.zero_grad()
            # with autocast(device_type='cuda', enabled=use_amp):  # Use autocast context
            #     outputs = model(input_ids=batch['input_ids'],
            #                     attention_mask=batch['attention_mask'],
            #                     labels=batch['labels'])
            #     loss = outputs.loss

            # if use_amp:
            #     scaler.scale(loss).backward()
            #     scaler.unscale_(optimizer)
            #     clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            #     loss.backward()
            #     clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     optimizer.step()
            if use_amp:
                with autocast("cuda"):
                    # print(model(input_ids=batch['input_ids'],
                    #                       attention_mask=batch['attention_mask'],
                    #                        labels=batch['labels']))
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"],
                    )
                    loss = outputs.loss
                    # if torch.isnan(loss) or torch.isinf(loss):
                    #     print("Warning: NaN or Inf loss detected!")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_train_loss += loss.item()
            # print(total_train_loss)

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=use_amp):
                    loss = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"],
                    ).loss
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Save if best
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
                print(f"⏹️ Early stopping (no improvement in {max_patience} epochs)")
                break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2E}"
        )
        csv_writer.writerow(
            [epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{current_lr:.2E}"]
        )
        csv_file.flush()

    csv_file.close()
    save_final_model_if_needed(model, save_dir, optimizer, scheduler)
    write_done_marker(save_dir)


# def train_for_finetune(model, train_loader, val_loader, cfg, save_dir, csv_file_path):
#     """
#     model        : a BartForConditionalGeneration
#     train_loader : DataLoader for pretrain set
#     val_loader   : DataLoader for validation set
#     cfg          : one of your hyperparameter dicts (e.g. hyperparameters_dict[key])
#     save_dir     : where to save the best model
#     csv_file_path: writes to the designated csv_file_path
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     # csv_path = os.path.join(save_dir, "training_log.csv")
#     # csv_file = open(csv_path, "w", newline="")
#     csv_file = open(csv_file_path, "w", newline="")
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
#     # if torch.cuda.device_count() > 1:
#     #    print("Let's use", torch.cuda.device_count(), "GPUs!")
#     #    model = nn.DataParallel(model)
#     model.to(device)
#
#     optimizer = make_optimizer(model, cfg)
#     print(f"length of train_loader: {len(train_loader)}")
#     print(f"Number of epochs: {cfg['TRAIN_EPOCHS']}")
#     scheduler = make_scheduler(optimizer, cfg, len(train_loader), cfg["TRAIN_EPOCHS"])
#
#     best_val_loss = float('inf')
#     # Check for checkpoint
#     checkpoint_path = os.path.join(save_dir, "checkpoint_resume.pt")
#     start_epoch = 1
#
#     if os.path.exists(checkpoint_path):
#         print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint["model_state"])
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#         if scheduler and checkpoint.get("scheduler_state"):
#             scheduler.load_state_dict(checkpoint["scheduler_state"])
#         best_val_loss = checkpoint["best_val_loss"]
#         patience = checkpoint["patience"]
#         start_epoch = checkpoint["epoch"] + 1
#     else:
#         # Back to config setup
#         patience = 0
#     thresh = cfg["early_stopping_threshold"]
#     max_patience = cfg["early_stopping_patience"]
#
#     use_amp = (gpu_used == "B200")
#     scaler = GradScaler("cuda") if use_amp else None
#
#     global_step = 0
#     batch_number = 0
#     for epoch in range(start_epoch, cfg["TRAIN_EPOCHS"] + 1):
#         # === Training ===
#         model.train()
#         total_train_loss = 0.0
#         step_in_epoch = 0
#
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
#             # print(batch)
#             # print(type(batch))  # print the type of batch itself
#             # for key,value in batch.items():
#             #     print(f"{key}: {value}")
#             batch = {k: v.to(device) for k, v in batch.items()}
#             # === Forward / Backward pass ===
#             if use_amp:
#                 with autocast("cuda"):
#                     outputs = model(input_ids=batch['input_ids'],
#                                     attention_mask=batch['attention_mask'],
#                                     labels=batch['labels'])
#                     loss = outputs.loss
#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
#                 clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 outputs = model(input_ids=batch['input_ids'],
#                                 attention_mask=batch['attention_mask'],
#                                 labels=batch['labels'])
#                 loss = outputs.loss
#                 loss.backward()
#                 clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad()
#
#             # ✅ Scheduler step after optimizer step and zero_grad
#             if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 scheduler.step()
#
#             total_train_loss += loss.item()
#             step_in_epoch += 1
#             global_step += 1
#
#             print(f"step_in_epoch: {step_in_epoch}\nglobal_step: {global_step}")
#
#             if global_step % validate_every_steps == 0:
#                 avg_train_loss = total_train_loss / step_in_epoch
#                 # avg_val_loss = run_validation(model, epoch, val_loader, device)
#
#                 # If batched
#                 avg_val_loss = run_validation_batched(model, epoch, val_loader, device,
#                                                       max_batches=MAX_VALID_BATCH_SIZE)
#
#                 # Save best checkpoint
#                 if avg_val_loss < best_val_loss - thresh:
#                     best_val_loss = avg_val_loss
#                     patience = 0
#                     model.save_pretrained(save_dir)
#                     checkpoint = {
#                         "epoch": epoch,
#                         "model_state": model.state_dict(),
#                         "optimizer_state": optimizer.state_dict(),
#                         "scheduler_state": scheduler.state_dict() if scheduler else None,
#                         "best_val_loss": best_val_loss,
#                         "patience": patience,
#                     }
#                     torch.save(checkpoint, checkpoint_path)
#                     print(f"✅ Checkpoint saved at step {global_step}")
#                 else:
#                     patience += 1
#                     if patience >= max_patience:
#                         print(f"Early stopping (no improvement in {max_patience} validation checks)")
#                         csv_file.close()
#                         save_final_model_if_needed(model, save_dir, optimizer, scheduler)
#                         write_done_marker(save_dir)
#                         return  # Early stop exit
#
#                 # Learning rate (use first group)
#                 current_lr = optimizer.param_groups[0]['lr']
#                 print(f"[Epoch {epoch} | Step {global_step}] train_loss={avg_train_loss:.4f}  "
#                       f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2E}")
#                 csv_writer.writerow(
#                     [epoch, global_step, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{current_lr:.2E}"])
#                 csv_file.flush()
#
#                 # batch_number += 1
#
#         # Reset train loss for the next validation interval
#         total_train_loss = 0.0
#         step_in_epoch = 0
#
#     # close the CSV file now that training (or early stop) is done
#     csv_file.close()
#
#     # Final model save (if needed) + mark training complete
#     save_final_model_if_needed(model, save_dir)
#     write_done_marker(save_dir)


def finetune_BART(hyperparameters_dict, args, key):
    """
    # TODO: 06/23/2025: FLESH THIS OUT
    Args:
        hyperparameters_dict:
        args:
        key:

    Returns:

    """
    # Setup File config parameters
    base_model_name = "skip_base"  # Customize as needed
    base_config = hyperparameters_dict[base_model_name]
    current_config = hyperparameters_dict[key]

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "./full_tokenizer_finetune_and_pretrain"
    )
    run_dir = f"./runs/selfies_BART_finetune_{key}"
    model_save_dir = os.path.join(run_dir, "model")
    csv_file_path = os.path.join(run_dir, "finetune_loss.csv")

    # Define Training Hyperparameters
    train_batch_size = current_config["TRAIN_BATCH_SIZE"]
    val_batch_size = current_config["VALID_BATCH_SIZE"]

    # Current embedding less than 512 one for finetune...
    df = pd.read_csv("./data/selfies_pretrain_w_tokencount.csv")

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

    finetune_train_dataset = SelfiesFinetuneDataset(
        dataframe=df_train, tokenizer=tokenizer, mode="finetune"
    )
    finetune_validation_dataset = SelfiesFinetuneDataset(
        dataframe=df_val, tokenizer=tokenizer, mode="finetune"
    )

    # Create DataLoader
    finetune_train_loader = DataLoader(
        finetune_train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, mode="fine"),
    )
    finetune_validation_loader = DataLoader(
        finetune_validation_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, mode="fine"),
    )

    # Single model path is the 7th one...
    model_path = "/home/kollin/Desktop/ExploreThesis/WIP_Thesis/runs/selfies_BART_PRETRAIN_model_small_lamb_earlyS_extradropout_highLR_lowThresh_or7th/model"
    # for inputs, labels in finetune_train_loader:
    #     print("Input shape:", inputs.shape)
    #     print("Input max", inputs.max())
    #     print("Input min", inputs.min())
    #     print("Label max", labels.max())
    #     print("Label min", labels.min())
    #     break
    #
    # for inputs, labels in finetune_validation_loader:
    #     print("Input shape:", inputs.shape)
    #     print("Input max", inputs.max())
    #     print("Input min", inputs.min())
    #     print("Label max", labels.max())
    #     print("Label min", labels.min())
    #     break

    for batch in finetune_train_loader:
        inputs = batch["input_ids"]
        labels = batch["labels"]  # pre-shifted labels with -100 for ignore
        print("Input shape:", inputs.shape)
        print("Input max", inputs.max())
        print("Input min", inputs.min())
        print("Label max", labels.max())
        print("Label min", labels.min())
        break

    for batch in finetune_validation_loader:
        inputs = batch["input_ids"]
        labels = batch["labels"]  # pre-shifted labels with -100 for ignore
        print("Input shape:", inputs.shape)
        print("Input max", inputs.max())
        print("Input min", inputs.min())
        print("Label max", labels.max())
        print("Label min", labels.min())
        break

    model = BartForConditionalGeneration.from_pretrained(model_path)
    # Freeze model except last decoder layer and LM head
    freeze_bart_layers(model, num_decoder_layers_unfrozen=1)

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name} | {param.numel()} params")

    for param in model.model.shared.parameters():
        param.requires_grad = False
    for param in model.model.encoder.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.model.decoder.embed_tokens.parameters():
        param.requires_grad = False

    # Test a forward pass
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(sf.split_selfies("[C][O][Si]"))]
    )

    # print(tokenizer.convert_ids_to_tokens(list(sf.split_selfies('[C][O][Si]'))))
    decoder_start_token_id = tokenizer.bos_token_id
    # output = model(input_ids=input_ids, decoder_input_ids=input_ids)
    # output = model(input_ids=input_ids, decoder_input_ids=input_ids, decoder_start_token_id=decoder_start_token_id)
    output = model(input_ids=input_ids, decoder_input_ids=input_ids)

    print("THIS")
    print("Model output shape:", output.logits.shape)

    train_for_finetune(
        model,
        finetune_train_loader,
        finetune_validation_loader,
        current_config,
        model_save_dir,
        csv_file_path,
    )


def main():
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
        help="Path of the SEFLIES dataset.",
    )
    parser.add_argument(
        "--subset_size",
        required=False,
        metavar="<int>",
        type=int,
        default=0,
        help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.",
    )
    parser.add_argument(
        "--hyperparameters_path",
        required=True,
        metavar="/path/to/hyperparameters/",
        help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.",
    )
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters_path)
    print("Loaded hyperparameters:", hyperparameters)
    bart_hyperparameters = hyperparameters.get(
        "BART", {}
    )  # This would have some model specific hyperparameters and probably be part of a for loop to iterate over the keys representing each model in the hyperparameters dictionary.
    print("BART hyperparameters:", bart_hyperparameters)

    for key in bart_hyperparameters.keys():
        if key.startswith("skip_"):
            continue

        run_dir = f"./runs/selfies_BART_finetune_{key}"

        # done_flag = os.path.join(run_dir, "done.txt")
        model_save_dir = os.path.join(run_dir, "model")
        done_flag = os.path.join(model_save_dir, "done.txt")

        if os.path.exists(done_flag):
            print(
                f"✅ Model '{key}' already completed (done.txt found) — skipping retrain."
            )
            continue
        else:
            args.smiles_dataset = f"model_name_{key}.csv"
            print(args.smiles_dataset)
            args.selfies_dataset = f"./data/molecule_data_{key}.csv"
            args.prepared_data_path = f"./data/prepared_selfies_{key}.txt"

            finetune_BART(bart_hyperparameters, args, key)


if __name__ == "__main__":
    main()

"""
TODO: Finetuning data requires the specific dataset from the beginning...
it needs to have site_name for it... My pretrain data does NOT need it...
"""

# TODO: I need to add early_stopping to the hyperparameters...
