import os
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_scheduler
import yaml
from pytorch_lamb import Lamb
import random
import selfies as sf
from torch.utils.data import Dataset
import torch_optimizer as optim

def diff_to_string(base_config, other_config):
    """

    Args:
        base_config:
        other_config:

    Returns:

    """
    diffs = {}
    for key in base_config:
        base_val = base_config.get(key)
        other_val = other_config.get(key)
        if base_val != other_val:
            diffs[key] = other_val
    return diffs

# def encode_differences_to_string(base_model_name, diffs):
def encode_differences_to_string(base_model_name, base_config, other_config):
    """
    Args:
        base_model_name:
        base_config:
        other_config:

    Returns:

    """
    diffs = diff_to_string(base_config, other_config)
    parts = [base_model_name]
    for key, value in sorted(diffs.items()):
        safe_key = str(key).upper().replace(" ", "_")
        safe_val = str(value).replace("/", "-").replace(" ", "").replace(":", "-")
        parts.append(f"{safe_key}-{safe_val}")
    return "__".join(parts)


def save_final_model_if_needed(model, save_dir):
    """Ensure the final model is saved if no best version was saved."""
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        print("🟡 No best model saved. Saving final model manually.")
        model.save_pretrained(save_dir)

def write_done_marker(save_dir):
    """Write a flag file to signal training completed."""
    done_path = os.path.join(save_dir, "done.txt")
    with open(done_path, "w") as f:
        f.write("Training complete\n")
    print(f"✅ Written done.txt to {done_path}")

def is_model_done(save_dir):
    """Check if training has already been completed for this model."""
    return os.path.exists(os.path.join(save_dir, "done.txt"))


def make_optimizer(model, cfg):
    lr = cfg["LEARNING_RATE"]
    opt = cfg["optimizer"].lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt == "lamb":
        return Lamb(model.parameters(), lr=lr)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    elif opt == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr)
    elif opt == "adafactor":
        return torch.optim.Adafactor(model.parameters(), lr=lr)
    # elif opt == ""
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def make_scheduler(optimizer, cfg, train_steps_per_epoch, num_epochs):
    lr_cfg = cfg.get("lr_sched", None)
    if not isinstance(lr_cfg, dict):
        return None

    sched_type = lr_cfg["type"].lower()
    warmup_steps = int(train_steps_per_epoch * num_epochs * lr_cfg.get("warmup_ratio", 0.0))
    print(f"Warmup ratio: {lr_cfg.get('warmup_ratio', 0.0)}")
    total_steps = train_steps_per_epoch * num_epochs

    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sched_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_cfg.get("step_size", 10),
                                               gamma=lr_cfg.get("gamma", 0.1))
    elif sched_type == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_cfg.get("milestones", [10, 20, 30]),
                                                    gamma=lr_cfg.get("gamma", 0.5))
    elif sched_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_cfg.get("gamma", 0.9))
    elif sched_type == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode="min",
                                                          patience=lr_cfg.get("patience", 3),
                                                          factor=lr_cfg.get("factor", 0.1),
                                                          verbose=True)
    elif sched_type == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                 base_lr=lr_cfg.get("base_lr", 1e-5),
                                                 max_lr=lr_cfg.get("max_lr", 1e-3),
                                                 step_size_up=lr_cfg.get("step_size_up", 5),
                                                 mode=lr_cfg.get("mode", "triangular2"),
                                                 cycle_momentum=False)
    elif sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=cfg["LEARNING_RATE"],
                                                   steps_per_epoch=train_steps_per_epoch,
                                                   epochs=num_epochs)
    elif sched_type == "inverse_sqrt":
        return get_scheduler(
            name="inverse_sqrt",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def make_scheduler_steps(optimizer, cfg, total_steps):
    lr_cfg = cfg.get("lr_sched", None)
    if not isinstance(lr_cfg, dict):
        return None

    sched_type = lr_cfg["type"].lower()
    warmup_steps = int(total_steps * lr_cfg.get("warmup_ratio", 0.0))
    print(f"Warmup ratio: {lr_cfg.get('warmup_ratio', 0.0)}")

    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sched_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_cfg.get("step_size", 10),
                                               gamma=lr_cfg.get("gamma", 0.1))
    elif sched_type == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_cfg.get("milestones", [10, 20, 30]),
                                                    gamma=lr_cfg.get("gamma", 0.5))
    elif sched_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_cfg.get("gamma", 0.9))
    elif sched_type == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode="min",
                                                          patience=lr_cfg.get("patience", 3),
                                                          factor=lr_cfg.get("factor", 0.1),
                                                          verbose=True)
    elif sched_type == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                 base_lr=lr_cfg.get("base_lr", 1e-5),
                                                 max_lr=lr_cfg.get("max_lr", 1e-3),
                                                 step_size_up=lr_cfg.get("step_size_up", 5),
                                                 mode=lr_cfg.get("mode", "triangular2"),
                                                 cycle_momentum=False)
    elif sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=cfg["LEARNING_RATE"],
                                                   total_steps=total_steps)
    elif sched_type == "inverse_sqrt":
        return get_scheduler(
            name="inverse_sqrt",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def load_hyperparameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

class ClusteredSelfiesDataset(Dataset):
    def __init__(self, df, tokenizer, mode='pretrain'):
        """
        Initializes the dataset with a clustered DataFrame and a tokenizer.

        Args:
            df (pandas.DataFrame): DataFrame containing at least a 'selfies' column.
                                     (It may also contain additional columns like 'Cluster'.)
            tokenizer: A tokenizer instance with an .encode() method.
            mode (str): Either 'pretrain' or 'finetune'. In 'finetune' mode, additional columns
                        (e.g., 'IC50' and 'site_name') are expected.
        """
        # Reset index to ensure integer indexing (0, 1, 2, ...)
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use .iloc to ensure row-based access.
        row = self.df.iloc[idx]
        selfies_string = row['selfies']
        encoded = self.tokenizer.encode(selfies_string)

        if self.mode == 'pretrain':
            return {'input_ids': torch.tensor(encoded.ids, dtype=torch.long)}
        # no finetune mode in this pretrain only portion...
                # elif self.mode == 'finetune':
                #     # Expect additional columns for fine-tuning.
                #     IC50 = row.get('IC50', 0)  # default value if missing
                #     inhibition_site = row.get('site_name', "")
                #     inhibition_encoded = self.encode_inhibition_site(inhibition_site)
                #     return {
                #         'input_ids': torch.tensor(encoded.ids, dtype=torch.long),
                #         'IC50': torch.tensor([IC50], dtype=torch.float),
                #         'inhibition_site': torch.tensor([inhibition_encoded], dtype=torch.long)
                #     }
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
